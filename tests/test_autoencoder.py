"""Pruebas de `src.autoencoder` con flujos sintéticos."""

import contextlib
import io
import json
import math
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
import pandas as pd
import torch

from src import autoencoder as ae
from src import config, etl
from src.config import (
    ATTACK_FAMILY_COLUMN,
    ATTACK_LABEL_COLUMN,
    BENIGN_LABEL,
    METADATA_COLUMNS,
    SPLITS,
    TARGET_COLUMN,
    TARGET_FPR_BUDGETS,
)

PORT = "Destination Port"
FEATURES = [PORT, "Flow Duration", "Flow Bytes/s", "IAT Mean", "Init_Win_bytes", "Pkts"]
FAST = {
    "epochs": 4,
    "batch_size": 128,
    "lr": 3e-3,
    "bottleneck_dim": 2,
    "device": "cpu",
}
PARAMS = ae.AutoencoderParams(**FAST)


def make_frame(rng, n_benign=1500, **attacks):
    """Benignos sobre una variedad de dimensión 1; ataques fuera de ella."""
    labels = [BENIGN_LABEL] * n_benign
    for name, n in attacks.items():
        labels += [name] * n
    labels = np.array(labels, dtype=object)
    attack = labels != BENIGN_LABEL
    n = labels.size
    # Un factor latente gobierna todas las features en el tráfico benigno.
    factor = rng.normal(size=n)
    noise = rng.normal(scale=0.05, size=(n, len(FEATURES)))
    broken = np.where(attack, -1.0, 1.0)
    values = np.column_stack(
        [
            np.expm1(np.abs(factor)) * 100,
            np.expm1(8 + factor + noise[:, 1]),
            np.expm1(6 + broken * factor + 6 * attack + noise[:, 2]),
            np.expm1(5 - factor + noise[:, 3]),
            np.where(factor > 0, 8192.0, -1.0),
            np.expm1(2 + 0.5 * broken * factor - 6 * attack + noise[:, 5]),
        ]
    )
    frame = pd.DataFrame(values, columns=FEATURES, dtype=np.float32)
    frame[TARGET_COLUMN] = attack.astype(np.int8)
    frame[ATTACK_FAMILY_COLUMN] = labels
    frame[ATTACK_LABEL_COLUMN] = labels
    frame["campaign_day"] = "synthetic"
    frame = frame[[*FEATURES, *METADATA_COLUMNS]]
    # Mezcladas: el filtro no puede depender de la posición.
    order = rng.permutation(n)
    return frame.iloc[order].reset_index(drop=True)


def make_splits(seed=0):
    rng = np.random.default_rng(seed)
    return {
        "train": make_frame(rng, DoS=200),
        "val": make_frame(rng, DoS=100),
        "test": make_frame(rng, DoS=100, Heartbleed=11, Bot=20),
    }


def tamper_attacks(frame):
    """Cambia las features de los ataques, con valores extremos."""
    frame = frame.copy()
    frame.loc[frame[TARGET_COLUMN] == 1, FEATURES] = np.float32(1e8)
    return frame


def drop_attacks(frame):
    return frame[frame[TARGET_COLUMN] == 0].reset_index(drop=True)


def write_artifacts(directory, splits, dataset="cicids2017"):
    """Persiste los splits con el formato que exige `etl.load_artifacts`."""
    paths = etl._artifact_paths(dataset, directory)
    for name in SPLITS:
        splits[name].to_parquet(paths[name], index=False)
    joblib.dump(etl.fit_benign_scaler(splits["train"]), paths["scaler"])
    metadata = {
        "schema_version": etl.SCHEMA_VERSION,
        "config_hash": "synthetic",
        "columns": {
            "features": config.get_feature_columns(splits["train"].columns),
            "metadata": list(METADATA_COLUMNS),
        },
        "counts": {name: {"total": len(splits[name])} for name in SPLITS},
        "artifacts": {
            name: {"sha256": etl.sha256_file(paths[name])}
            for name in (*SPLITS, "scaler")
        },
    }
    paths["metadata"].write_text(json.dumps(metadata), encoding="utf-8")


def thresholds_of(report):
    return {row["max_fpr"]: row["threshold"] for row in report["budgets"]}


class SignedLog1pTests(unittest.TestCase):
    def test_compresses_extreme_tails(self):
        out = ae.signed_log1p(np.array([1e8, -1e8, 2.071e9]))
        self.assertLess(np.abs(out).max(), 22)
        self.assertAlmostEqual(out[0], math.log1p(1e8))
        self.assertEqual(out[1], -out[0])

    def test_preserves_negative_sentinel(self):
        out = ae.signed_log1p(np.array([-1.0, 0.0, 1.0]))
        np.testing.assert_allclose(out, [-math.log(2), 0.0, math.log(2)])

    def test_is_monotonic_and_float64(self):
        x = np.array([-1e8, -5, -1, -0.5, 0, 0.5, 1, 5, 1e8], dtype=np.float32)
        out = ae.signed_log1p(x)
        self.assertEqual(out.dtype, np.float64)
        self.assertTrue((np.diff(out) > 0).all())


class PreprocessorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = make_splits()["train"]
        cls.pre = ae.FlowAEPreprocessor.fit(cls.train, FEATURES, 10.0)

    def test_fitted_on_benign_train_only(self):
        benign = self.train.loc[self.train[TARGET_COLUMN] == 0, FEATURES]
        logs = ae.signed_log1p(benign.to_numpy())
        np.testing.assert_allclose(self.pre.mean, logs.mean(axis=0), rtol=1e-9)
        np.testing.assert_allclose(self.pre.scale, logs.std(axis=0), rtol=1e-9)
        # Control: incluir los ataques daría otra media.
        everything = ae.signed_log1p(self.train[FEATURES].to_numpy())
        self.assertFalse(np.allclose(self.pre.mean, everything.mean(axis=0)))

    def test_attacks_do_not_change_a_bit(self):
        for change in (tamper_attacks, drop_attacks):
            other = ae.FlowAEPreprocessor.fit(change(self.train), FEATURES, 10.0)
            np.testing.assert_array_equal(other.mean, self.pre.mean)
            np.testing.assert_array_equal(other.scale, self.pre.scale)

    def test_clipping(self):
        frame = self.train.head(2).copy()
        frame.loc[0, "Pkts"] = np.float32(1e30)
        frame.loc[1, "Pkts"] = np.float32(-1e30)
        z = ae.FlowAEPreprocessor.fit(self.train, FEATURES, 3.0).transform(frame)
        self.assertEqual(z[0, -1], 3.0)
        self.assertEqual(z[1, -1], -3.0)
        self.assertLessEqual(np.abs(z).max(), 3.0)
        self.assertEqual(z.dtype, np.float32)

    def test_transform_does_not_depend_on_chunks(self):
        expected = self.pre.transform(self.train)
        with mock.patch.object(ae, "CHUNK_ROWS", 7):
            np.testing.assert_array_equal(self.pre.transform(self.train), expected)

    def test_non_finite_is_rejected(self):
        for bad in (np.nan, np.inf):
            frame = self.train.head(3).copy()
            frame.loc[1, "Pkts"] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.pre.transform(frame)

    def test_missing_feature_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Pkts"):
            self.pre.transform(self.train.drop(columns="Pkts"))

    def test_constant_feature_gives_no_nan(self):
        train = self.train.copy()
        train["Pkts"] = np.float32(3)
        z = ae.FlowAEPreprocessor.fit(train, FEATURES, 10.0).transform(train)
        self.assertTrue(np.isfinite(z).all())

    def test_needs_benign(self):
        attacks = self.train[self.train[TARGET_COLUMN] == 1]
        with self.assertRaises(ValueError):
            ae.FlowAEPreprocessor.fit(attacks, FEATURES, 10.0)

    def test_state_round_trip(self):
        again = ae.FlowAEPreprocessor.from_state_dict(self.pre.state_dict())
        np.testing.assert_array_equal(
            again.transform(self.train), self.pre.transform(self.train)
        )
        state = {**self.pre.state_dict(), "feature_cols": FEATURES[1:]}
        with self.assertRaises(etl.ArtifactError):
            ae.FlowAEPreprocessor.from_state_dict(state)


class ModelTests(unittest.TestCase):
    def test_architecture(self):
        model = ae.FlowAutoencoder(70, bottleneck_dim=16)
        linear = [
            (m.in_features, m.out_features)
            for m in model.modules()
            if isinstance(m, torch.nn.Linear)
        ]
        self.assertEqual(
            linear, [(70, 64), (64, 32), (32, 16), (16, 32), (32, 64), (64, 70)]
        )
        kinds = [type(m) for m in model.modules()]
        self.assertEqual(kinds.count(torch.nn.LayerNorm), 4)
        self.assertEqual(kinds.count(torch.nn.SiLU), 4)
        self.assertEqual(kinds.count(torch.nn.Dropout), 4)
        self.assertNotIn(torch.nn.BatchNorm1d, kinds)
        self.assertIsInstance(model.decoder[-1], torch.nn.Linear)
        self.assertEqual(model(torch.zeros(1, 70)).shape, (1, 70))

    def test_score_metrics(self):
        class Zero(torch.nn.Module):
            def forward(self, x):
                return torch.zeros_like(x)

        flows = np.array([[3.0, -1.0], [0.5, 0.0]], dtype=np.float32)
        for metric, expected in (
            ("mse", [5.0, 0.125]),
            ("mae", [2.0, 0.25]),
            ("huber", [1.5, 0.0625]),
        ):
            with self.subTest(metric=metric):
                scores = ae.reconstruction_error(Zero(), flows, metric)
                self.assertEqual(scores.dtype, np.float64)
                np.testing.assert_allclose(scores, expected)
        with self.assertRaises(ValueError):
            ae.reconstruction_error(Zero(), flows, "otra")

    def test_scoring_leaves_the_model_untouched(self):
        model = ae.FlowAutoencoder(4).train()
        ae.reconstruction_error(model, np.ones((2, 4), dtype=np.float32))
        self.assertTrue(model.training)
        self.assertEqual(next(model.parameters()).dtype, torch.float32)

    def test_score_is_deterministic_and_chunk_independent(self):
        torch.manual_seed(0)
        model = ae.FlowAutoencoder(6).train()  # el dropout no debe actuar
        flows = np.random.default_rng(0).normal(size=(300, 6)).astype(np.float32)
        expected = ae.reconstruction_error(model, flows)
        np.testing.assert_array_equal(ae.reconstruction_error(model, flows), expected)
        with mock.patch.object(ae, "CHUNK_ROWS", 7):
            chunked = ae.reconstruction_error(model, flows)
        np.testing.assert_allclose(chunked, expected, rtol=1e-9)

    def test_non_finite_scores_are_rejected(self):
        model = ae.FlowAutoencoder(2)
        with torch.no_grad():
            model.decoder[-1].bias.fill_(math.nan)
        with self.assertRaises(ValueError):
            ae.reconstruction_error(model, np.ones((1, 2), dtype=np.float32))


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splits = make_splits()
        cls.model, cls.pre, cls.info = cls.fit(cls.splits)

    @staticmethod
    def fit(splits, params=PARAMS):
        return ae.fit(splits["train"], splits["val"], FEATURES, params)

    def assert_same_weights(self, model):
        expected = self.model.state_dict()
        for name, tensor in model.state_dict().items():
            self.assertTrue(torch.equal(tensor, expected[name]), name)

    def test_attacks_never_touch_weights_scaler_or_threshold(self):
        def thresholds(splits, model, pre):
            def score(frame):
                return ae.reconstruction_error(model, pre.transform(frame))

            return thresholds_of(ae.evaluate(splits["val"], splits["test"], score))

        expected = thresholds(self.splits, self.model, self.pre)
        for change in (tamper_attacks, drop_attacks):
            with self.subTest(change=change.__name__):
                splits = {
                    "train": change(self.splits["train"]),
                    "val": change(self.splits["val"]),
                    "test": tamper_attacks(self.splits["test"]),
                }
                model, pre, info = self.fit(splits)
                self.assert_same_weights(model)
                np.testing.assert_array_equal(pre.mean, self.pre.mean)
                np.testing.assert_array_equal(pre.scale, self.pre.scale)
                self.assertEqual(info["history"], self.info["history"])
                self.assertEqual(thresholds(splits, model, pre), expected)

    def test_benign_rows_do_matter(self):
        # Control del anterior: el mismo cambio sobre benignos sí se nota.
        splits = dict(self.splits)
        train = splits["train"].copy()
        first = train.index[train[TARGET_COLUMN] == 0][0]
        train.loc[first, FEATURES] = np.float32(1e8)
        model, pre, _ = self.fit({**splits, "train": train})
        self.assertFalse(np.array_equal(pre.mean, self.pre.mean))

        val = splits["val"].copy()
        val.loc[val[TARGET_COLUMN] == 0, "Pkts"] *= np.float32(3)
        _, _, info = self.fit({**splits, "val": val})
        self.assertNotEqual(
            info["history"][0]["val_loss"], self.info["history"][0]["val_loss"]
        )

    def test_only_benign_rows_reach_the_trainer(self):
        with mock.patch.object(ae, "train_autoencoder", return_value=(0, 0)) as spy:
            ae.fit(self.splits["train"], self.splits["val"], FEATURES, PARAMS)
        x_train, x_val, _ = spy.call_args.args
        for name, matrix in (("train", x_train), ("val", x_val)):
            benign = drop_attacks(self.splits[name])
            np.testing.assert_array_equal(matrix, self.pre.transform(benign))

    def test_same_seed_same_weights(self):
        model, _, info = self.fit(self.splits)
        self.assert_same_weights(model)
        self.assertEqual(info["history"], self.info["history"])
        other, _, _ = self.fit(
            self.splits, ae.AutoencoderParams(**FAST, random_state=1)
        )
        with self.assertRaises(AssertionError):
            self.assert_same_weights(other)

    def test_loss_improves(self):
        history = self.info["history"]
        self.assertLess(history[-1]["val_loss"], history[0]["val_loss"])
        self.assertTrue(all(math.isfinite(row["train_loss"]) for row in history))

    def test_out_of_distribution_attack_scores_higher(self):
        test = self.splits["test"]
        scores = ae.reconstruction_error(self.model, self.pre.transform(test))
        attack = test[TARGET_COLUMN].to_numpy() == 1
        self.assertGreater(np.median(scores[attack]), 3 * np.median(scores[~attack]))
        ranking = ae.rank_attacks(test[ATTACK_LABEL_COLUMN], scores)
        self.assertEqual(set(ranking), {"DoS", "Heartbleed", "Bot"})
        for row in ranking.values():
            self.assertGreater(row["roc_auc"], 0.95)

    def run_with_val_losses(self, losses, **kwargs):
        """Entrena con pérdidas de Val prefijadas; registra los pesos por época."""
        states, pending = [], iter(losses)

        def fake(model, data, criterion):
            states.append({k: v.clone() for k, v in model.state_dict().items()})
            return next(pending)

        x = np.random.default_rng(0).normal(size=(64, 4)).astype(np.float32)
        params = ae.AutoencoderParams(**{**FAST, **kwargs})
        with mock.patch.object(ae, "_mean_loss", fake):
            model, info = ae.train_autoencoder(x, x, params)
        return model, info, states

    def test_early_stopping_restores_the_best_epoch(self):
        model, info, states = self.run_with_val_losses(
            [5.0, 1.0, 2.0, 3.0, 0.1], epochs=5, patience=2
        )
        self.assertEqual(
            (info["best_epoch"], info["epochs_run"], info["stopped_early"]),
            (2, 4, True),
        )
        self.assertEqual(info["best_val_loss"], 1.0)
        for name, tensor in model.state_dict().items():
            self.assertTrue(torch.equal(tensor, states[1][name]))
        self.assertFalse(model.training)

    def test_plateau_reduces_learning_rate(self):
        _, info, _ = self.run_with_val_losses([1.0] * 8, epochs=8, patience=20)
        rates = [row["lr"] for row in info["history"]]
        self.assertEqual(rates[0], FAST["lr"])
        self.assertEqual(rates[-1], FAST["lr"] * ae.LR_FACTOR)
        self.assertFalse(info["stopped_early"])

    def test_gradients_are_clipped(self):
        x = np.ones((16, 4), dtype=np.float32)
        with mock.patch.object(
            ae.nn.utils, "clip_grad_norm_", wraps=ae.nn.utils.clip_grad_norm_
        ) as spy:
            ae.train_autoencoder(x, x, ae.AutoencoderParams(**FAST))
        self.assertGreater(spy.call_count, 0)
        self.assertEqual(spy.call_args.args[1], 1.0)

    def test_non_finite_loss_aborts(self):
        with self.assertRaises(RuntimeError):
            self.run_with_val_losses([math.nan])

    def test_single_row_batch(self):
        x = np.random.default_rng(0).normal(size=(129, 4)).astype(np.float32)
        _, info = ae.train_autoencoder(x, x, PARAMS)  # último batch: 1 fila
        self.assertTrue(math.isfinite(info["best_val_loss"]))

    def test_losses(self):
        x = np.full((8, 2), 3.0, dtype=np.float32)
        for name, kind in (("mse", "MSELoss"), ("smooth_l1", "SmoothL1Loss")):
            self.assertEqual(type(ae._make_loss(name)).__name__, kind)
            _, info = ae.train_autoencoder(
                x, x, ae.AutoencoderParams(**FAST, loss=name)
            )
            self.assertTrue(math.isfinite(info["best_val_loss"]))


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.dirs = {
            name: self.root / name
            for name in ("processed_dir", "models_dir", "metrics_dir")
        }
        for path in self.dirs.values():
            path.mkdir()
        self.splits = make_splits()
        write_artifacts(self.dirs["processed_dir"], self.splits)

    def run_ae(self, params=PARAMS, **kwargs):
        return ae.run(params=params, **self.dirs, **kwargs)

    def load(self, suffix=""):
        return ae.load_autoencoder(
            suffix=suffix,
            models_dir=self.dirs["models_dir"],
            metrics_dir=self.dirs["metrics_dir"],
        )

    def test_exact_round_trip(self):
        model, pre, _ = ae.fit(
            self.splits["train"], self.splits["val"], FEATURES, PARAMS
        )
        path = ae.save_checkpoint(self.root / "x.pt", model, pre, "mae")
        loaded, loaded_pre, rest = ae.load_checkpoint(path)
        self.assertEqual(rest["score_metric"], "mae")
        self.assertEqual(loaded_pre.feature_cols, FEATURES)
        self.assertEqual(loaded_pre.clip_z, pre.clip_z)
        np.testing.assert_array_equal(loaded_pre.mean, pre.mean)
        np.testing.assert_array_equal(loaded_pre.scale, pre.scale)
        for name, tensor in loaded.state_dict().items():
            self.assertTrue(torch.equal(tensor, model.state_dict()[name]))

        test = self.splits["test"]
        scores = ae.score_dataframe(path, test, device="cpu")
        self.assertEqual(scores.dtype, np.float64)
        self.assertEqual(scores.shape, (len(test),))
        self.assertTrue(np.isfinite(scores).all())
        np.testing.assert_array_equal(
            scores, ae.reconstruction_error(model, pre.transform(test), "mae")
        )

    def test_score_dataframe_uses_the_saved_metric(self):
        model, pre, _ = ae.fit(
            self.splits["train"], self.splits["val"], FEATURES, PARAMS
        )
        test = self.splits["test"]
        for metric in ae.SCORE_METRICS:
            with self.subTest(metric=metric):
                path = ae.save_checkpoint(self.root / "x.pt", model, pre, metric)
                np.testing.assert_array_equal(
                    ae.score_dataframe(path, test, device="cpu"),
                    ae.reconstruction_error(model, pre.transform(test), metric),
                )

    def test_score_dataframe_needs_only_features(self):
        self.run_ae()
        path, _ = self.load()
        test = self.splits["test"]
        expected = ae.score_dataframe(str(path), test, device="cpu")
        shuffled = test[FEATURES[::-1]].set_index(np.arange(len(test)) * 3 + 7)
        np.testing.assert_array_equal(
            ae.score_dataframe(path, shuffled, device="cpu"), expected
        )
        np.testing.assert_allclose(
            ae.score_dataframe(path, test.iloc[::-1], device="cpu")[::-1],
            expected,
            rtol=1e-9,
        )
        empty = ae.score_dataframe(path, test.head(0), device="cpu")
        self.assertEqual((empty.shape, empty.dtype), ((0,), np.float64))
        with self.assertRaises(ValueError):
            ae.score_dataframe(path, test.drop(columns="Pkts"), device="cpu")

    def test_report_matches_checkpoint(self):
        report = self.run_ae()
        path, saved = self.load()
        self.assertEqual(saved, report)
        self.assertEqual(path.name, "autoencoder_cicids2017.pt")
        test = self.splits["test"]
        scores = ae.score_dataframe(path, test, device="cpu")
        attack = test[TARGET_COLUMN].to_numpy() == 1
        self.assertEqual(tuple(thresholds_of(saved)), TARGET_FPR_BUDGETS)
        for row in saved["budgets"]:
            hits = (scores >= row["threshold"]) & attack
            self.assertEqual(int(hits.sum()), row["test"]["confusion_matrix"]["tp"])
            self.assertLessEqual(row["val"]["fpr"], row["max_fpr"])
            self.assertEqual(
                row["test_fpr_within_budget"], row["test"]["fpr"] <= row["max_fpr"]
            )
            self.assertEqual(set(row["focus_attacks"]), {"Heartbleed", "Bot"})
            for column in (ATTACK_FAMILY_COLUMN, ATTACK_LABEL_COLUMN):
                per_attack = row[f"by_{column}"]["per_attack"]
                self.assertEqual(set(per_attack), {"DoS", "Heartbleed", "Bot"})
        history = saved["training"]["history"]
        self.assertEqual(len(history), saved["training"]["epochs_run"])
        self.assertEqual(saved["training"]["n_train_benign"], 1500)
        self.assertEqual(saved["training"]["n_val_benign"], 1500)
        self.assertIn("Heartbleed *", ae.format_report(report))

    def test_attacks_do_not_change_the_saved_thresholds(self):
        first = self.run_ae()
        for name in SPLITS:
            self.splits[name] = tamper_attacks(self.splits[name])
        write_artifacts(self.dirs["processed_dir"], self.splits)
        second = self.run_ae(tag="alterado")
        self.assertEqual(thresholds_of(first), thresholds_of(second))
        self.assertNotEqual(
            first["ranking_by_label_raw"], second["ranking_by_label_raw"]
        )

    def test_val_without_attacks(self):
        self.splits["val"] = drop_attacks(self.splits["val"])
        write_artifacts(self.dirs["processed_dir"], self.splits)
        report = self.run_ae()
        self.assertIsNone(report["budgets"][0]["val"]["roc_auc"])
        ae.format_report(report)

    def test_report_is_strict_json_without_nan(self):
        def reject(constant):
            raise AssertionError(f"{constant} en el reporte")

        self.run_ae()
        path = self.dirs["metrics_dir"] / "autoencoder_cicids2017.json"
        report = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
        self.assertEqual(report["score_dtype"], "float64")

    def test_drop_dst_port(self):
        report = self.run_ae(drop_dst_port=True)
        self.assertEqual(report["dropped_features"], [PORT])
        path, _ = self.load("_no_dst_port")
        _, pre, _ = ae.load_checkpoint(path)
        self.assertEqual(pre.feature_cols, FEATURES[1:])
        test = self.splits["test"]
        changed = test.copy()
        changed[PORT] = np.float32(9999)
        np.testing.assert_array_equal(
            ae.score_dataframe(path, changed, device="cpu"),
            ae.score_dataframe(path, test.drop(columns=PORT), device="cpu"),
        )

    def test_drop_dst_port_without_port(self):
        for name in SPLITS:
            self.splits[name] = self.splits[name].drop(columns=PORT)
        write_artifacts(self.dirs["processed_dir"], self.splits)
        with self.assertRaises(ValueError):
            self.run_ae(drop_dst_port=True)

    def test_variants_do_not_overwrite(self):
        self.run_ae()
        path = self.dirs["metrics_dir"] / "autoencoder_cicids2017.json"
        before = path.read_bytes()
        self.run_ae(tag="v2", drop_dst_port=True)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(
            sorted(p.name for p in self.dirs["models_dir"].iterdir()),
            ["autoencoder_cicids2017.pt", "autoencoder_cicids2017_v2_no_dst_port.pt"],
        )

    def test_missing_or_tampered(self):
        with self.assertRaises(etl.ArtifactError):
            self.load()
        with self.assertRaises(etl.ArtifactError):
            ae.load_checkpoint(self.root / "nada.pt")
        self.run_ae()
        path, _ = self.load()
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaises(etl.ArtifactError):
            self.load()
        path.unlink()
        with self.assertRaises(etl.ArtifactError):
            self.load()

    def test_other_version_is_rejected(self):
        for payload in ({"version": "0.0.1"}, [1, 2]):
            path = self.root / "viejo.pt"
            torch.save(payload, path)
            with self.subTest(payload=payload), self.assertRaises(etl.ArtifactError):
                ae.load_checkpoint(path)

    def test_code_in_checkpoint_is_not_executed(self):
        marker = self.root / "ejecutado"

        class Malicious:
            def __reduce__(self):
                return Path.touch, (marker,)

        path = self.root / "malo.pt"
        torch.save({"version": ae.CHECKPOINT_VERSION, "x": Malicious()}, path)
        with self.assertRaises(pickle.UnpicklingError):
            ae.load_checkpoint(path)
        self.assertFalse(marker.exists())

    def test_default_paths_follow_config(self):
        self.assertEqual(
            config.get_autoencoder_checkpoint_path("cicids2017", "_v2"),
            config.MODELS_DIR / "autoencoder_cicids2017_v2.pt",
        )
        self.assertEqual(
            config.get_autoencoder_report_path("cicids2017"),
            config.METRICS_DIR / "autoencoder_cicids2017.json",
        )
        with self.assertRaises(ValueError):
            config.get_autoencoder_checkpoint_path("otro")


class ParamsAndCliTests(unittest.TestCase):
    def test_invalid_params(self):
        for bad in (
            {"epochs": 0},
            {"batch_size": 0},
            {"patience": 0},
            {"bottleneck_dim": 0},
            {"lr": 0.0},
            {"clip_z": math.inf},
            {"dropout": 1.0},
            {"loss": "otra"},
            {"score_metric": "otra"},
            {"device": "tpu"},
        ):
            with self.subTest(**bad), self.assertRaises(ValueError):
                ae.AutoencoderParams(**bad)

    def test_device_resolution(self):
        for available, expected in ((True, "cuda"), (False, "cpu")):
            with mock.patch.object(
                torch.cuda, "is_available", return_value=available
            ):
                self.assertEqual(ae.resolve_device("auto"), expected)
                self.assertEqual(ae.resolve_device("cpu"), "cpu")
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=False),
            self.assertRaises(ValueError),
        ):
            ae.resolve_device("cuda")
        with self.assertRaises(ValueError):
            ae.resolve_device("tpu")

    def test_does_not_need_the_supervised_backends(self):
        code = (
            "import sys; sys.modules['xgboost'] = sys.modules['lightgbm'] = None; "
            "import src.autoencoder"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=config.PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def invoke(self, *argv):
        with (
            mock.patch.object(ae, "run") as run,
            mock.patch.object(ae, "ensure_directories"),
            mock.patch.object(ae, "format_report", return_value=""),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            ae.main(list(argv))
        return run.call_args.args

    def test_defaults(self):
        self.assertEqual(
            self.invoke(), ("cicids2017", ae.AutoencoderParams(), False, "")
        )
        self.assertEqual(ae.AutoencoderParams().batch_size, 1024)

    def test_flags(self):
        args = self.invoke(
            *("--epochs", "3", "--batch-size", "64", "--lr", "0.01"),
            *("--patience", "2", "--bottleneck-dim", "16", "--clip-z", "5"),
            *("--loss", "smooth_l1", "--score-metric", "huber"),
            *("--device", "cpu", "--drop-dst-port", "--tag", "v2"),
        )
        params = ae.AutoencoderParams(
            epochs=3,
            batch_size=64,
            lr=0.01,
            patience=2,
            bottleneck_dim=16,
            clip_z=5.0,
            loss="smooth_l1",
            score_metric="huber",
            device="cpu",
        )
        self.assertEqual(args, ("cicids2017", params, True, "v2"))

    def test_invalid_flags(self):
        for argv in (
            ["--epochs", "0"],
            ["--clip-z", "-1"],
            ["--loss", "otra"],
            ["--score-metric", "otra"],
            ["--tag", "../x"],
        ):
            with self.subTest(argv=argv), self.assertRaises(SystemExit) as caught:
                self.invoke(*argv)
            self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
