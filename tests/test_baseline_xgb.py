"""Pruebas de `src.baseline_xgb` con flujos sintéticos."""

import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
import pandas as pd

from src import baseline_xgb as baseline
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
FEATURES = [PORT, "Flow Duration", "Flow Bytes/s", "Noise"]
FAMILIES = {"SSH-Patator": "BruteForce"}
FAST = {"n_estimators": 25, "max_depth": 3, "learning_rate": 0.3, "device": "cpu"}


def make_frame(rng, n_benign=1500, **attacks):
    """Benignos y ataques solapados en duración y tasa."""
    labels = [BENIGN_LABEL] * n_benign
    for name, n in attacks.items():
        labels += [name.replace("_", "-")] * n
    labels = np.array(labels, dtype=object)
    attack = labels != BENIGN_LABEL
    n = labels.size
    frame = pd.DataFrame(
        {
            PORT: np.where(attack, 22, rng.choice([22, 53, 80, 443], n)),
            "Flow Duration": rng.normal(np.where(attack, 1.5, 0.0), 1.0),
            "Flow Bytes/s": rng.normal(np.where(attack, 1.0, 0.0), 1.0),
            "Noise": rng.normal(size=n),
        },
        dtype=np.float32,
    )
    frame[TARGET_COLUMN] = attack.astype(np.int8)
    frame[ATTACK_FAMILY_COLUMN] = [FAMILIES.get(x, x) for x in labels]
    frame[ATTACK_LABEL_COLUMN] = labels
    frame["campaign_day"] = "synthetic"
    return frame[[*FEATURES, *METADATA_COLUMNS]]


def make_splits(seed=0):
    rng = np.random.default_rng(seed)
    splits = {
        "train": make_frame(rng, PortScan=300, SSH_Patator=200),
        "val": make_frame(rng, PortScan=150, SSH_Patator=100),
        "test": make_frame(rng, PortScan=150, SSH_Patator=100, Heartbleed=10),
    }
    # 40 PortScan de Test copian vectores exactos de Train.
    test, train = splits["test"], splits["train"]
    targets = test.index[test[ATTACK_LABEL_COLUMN] == "PortScan"][:40]
    sources = train.index[train[ATTACK_LABEL_COLUMN] == "PortScan"][:40]
    test.loc[targets, FEATURES] = train.loc[sources, FEATURES].to_numpy()
    return splits


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


def flip_test(splits):
    """Invierte todas las etiquetas de Test; Train y Val quedan intactos."""
    test = splits["test"].copy()
    was_attack = test[TARGET_COLUMN] == 1
    test[TARGET_COLUMN] = (~was_attack).astype(np.int8)
    label = np.where(was_attack, BENIGN_LABEL, "PortScan")
    test[ATTACK_FAMILY_COLUMN] = label
    test[ATTACK_LABEL_COLUMN] = label
    return {**splits, "test": test}


def thresholds_of(report):
    return {row["max_fpr"]: row["threshold"] for row in report["budgets"]}


def non_finite(value, path="$"):
    """Rutas de los flotantes NaN o Inf dentro de un JSON."""
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in non_finite(v, f"{path}.{k}")]
    if isinstance(value, list):
        return [p for i, v in enumerate(value) for p in non_finite(v, f"{path}[{i}]")]
    return [path] if isinstance(value, float) and not math.isfinite(value) else []


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splits = make_splits()
        cls.params = baseline.TrainParams(**FAST)
        cls.model, cls.report = baseline.fit_and_evaluate(cls.splits, cls.params)

    def test_test_labels_do_not_change_thresholds_or_model(self):
        model, report = baseline.fit_and_evaluate(flip_test(self.splits), self.params)
        self.assertEqual(thresholds_of(report), thresholds_of(self.report))
        self.assertEqual(
            model.booster.save_raw("json"), self.model.booster.save_raw("json")
        )
        # Control: la evaluación sí depende de esas etiquetas.
        self.assertNotEqual(
            report["budgets"][0]["test"]["recall"],
            self.report["budgets"][0]["test"]["recall"],
        )

    def test_test_features_do_not_change_thresholds(self):
        test = self.splits["test"].copy()
        test[FEATURES] = test[FEATURES] + np.float32(5)
        _, report = baseline.fit_and_evaluate(
            {**self.splits, "test": test}, self.params
        )
        self.assertEqual(thresholds_of(report), thresholds_of(self.report))

    def test_val_labels_do_change_thresholds(self):
        val = self.splits["val"]
        scores = baseline.score_supervised(self.model, val)
        y = val[TARGET_COLUMN].to_numpy()
        self.assertNotEqual(
            baseline.calibrate_thresholds(y, scores),
            baseline.calibrate_thresholds(1 - y, scores),
        )

    def test_every_budget_holds_in_val(self):
        self.assertEqual(tuple(thresholds_of(self.report)), TARGET_FPR_BUDGETS)
        for row in self.report["budgets"]:
            self.assertLessEqual(row["val"]["fpr"], row["max_fpr"])
            self.assertEqual(
                row["test_fpr_within_budget"], row["test"]["fpr"] <= row["max_fpr"]
            )

    def test_scores_are_float64_probabilities(self):
        scores = baseline.score_supervised(self.model, self.splits["test"])
        self.assertEqual(scores.dtype, np.float64)
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(((scores >= 0) & (scores <= 1)).all())
        self.assertGreater(self.report["budgets"][0]["test"]["roc_auc"], 0.8)

    def test_input_frames_are_not_modified(self):
        pristine = make_splits()
        for name in SPLITS:
            pd.testing.assert_frame_equal(self.splits[name], pristine[name])

    def test_unseen_attacks_are_reported_apart(self):
        unseen = self.report["unseen_in_train"]
        self.assertEqual(unseen[ATTACK_LABEL_COLUMN], ["Heartbleed"])
        for row in self.report["budgets"]:
            for column in (ATTACK_FAMILY_COLUMN, ATTACK_LABEL_COLUMN):
                groups = row[f"by_{column}"]
                self.assertEqual(list(groups["unseen"]), ["Heartbleed"])
                self.assertEqual(groups["unseen"]["Heartbleed"]["n_train"], 0)
                self.assertNotIn("Heartbleed", groups["seen"])
                recalls = [r["recall"] for r in groups["seen"].values()]
                self.assertAlmostEqual(groups["macro_recall_seen"], np.mean(recalls))

    def test_unseen_label_of_a_seen_family(self):
        test = self.splits["test"].copy()
        mask = test[ATTACK_LABEL_COLUMN] == "Heartbleed"
        test.loc[mask, ATTACK_FAMILY_COLUMN] = "PortScan"
        _, report = baseline.fit_and_evaluate(
            {**self.splits, "test": test}, self.params
        )
        self.assertEqual(report["unseen_in_train"][ATTACK_LABEL_COLUMN], ["Heartbleed"])
        self.assertEqual(report["unseen_in_train"][ATTACK_FAMILY_COLUMN], [])

    def test_top_features(self):
        top = self.report["top_features_by_gain"]
        gains = [row["gain"] for row in top]
        self.assertEqual(gains, sorted(gains, reverse=True))
        self.assertLessEqual({row["feature"] for row in top}, set(FEATURES))
        self.assertEqual(len(baseline.top_features(self.model, limit=2)), 2)


class TwinTests(unittest.TestCase):
    def frame(self, rows):
        return pd.DataFrame(rows, columns=["a", "b"], dtype=np.float32)

    def test_exact_detection(self):
        train = self.frame([[1, 2], [3, 4], [3, 4]])
        flows = self.frame([[1, 2], [2, 1], [1, 4], [3, 4]])
        self.assertEqual(
            baseline.find_train_twins(train, flows, ["a", "b"]).tolist(),
            [True, False, False, True],
        )

    def test_one_ulp_is_not_a_twin(self):
        train = self.frame([[1, 2]])
        flows = self.frame([[1, 2]])
        flows.loc[0, "b"] = np.nextafter(np.float32(2), np.float32(3))
        self.assertFalse(baseline.find_train_twins(train, flows, ["a", "b"]).any())

    def test_negative_zero_equals_zero(self):
        train = self.frame([[0.0, 1]])
        flows = self.frame([[-0.0, 1]])
        self.assertTrue(baseline.find_train_twins(train, flows, ["a", "b"]).all())

    def test_hash_collision_is_not_a_twin(self):
        train = self.frame([[1, 2], [5, 6]])
        flows = self.frame([[1, 2], [7, 8]])

        def constant(frame, index):
            return pd.Series(np.zeros(len(frame), dtype=np.uint64))

        with mock.patch.object(pd.util, "hash_pandas_object", constant):
            twins = baseline.find_train_twins(train, flows, ["a", "b"])
        self.assertEqual(twins.tolist(), [True, False])

    def test_only_model_features_count(self):
        train = self.frame([[1, 2]])
        flows = self.frame([[1, 9]])
        self.assertFalse(baseline.find_train_twins(train, flows, ["a", "b"]).any())
        self.assertTrue(baseline.find_train_twins(train, flows, ["a"]).all())

    def test_matches_brute_force(self):
        rng = np.random.default_rng(1)
        train = self.frame(rng.integers(0, 12, (60, 2)))
        flows = self.frame(rng.integers(0, 12, (200, 2)))
        known = set(map(tuple, train.to_numpy()))
        expected = [tuple(row) in known for row in flows.to_numpy()]
        twins = baseline.find_train_twins(train, flows, ["a", "b"])
        self.assertEqual(twins.tolist(), expected)
        self.assertTrue(0 < twins.sum() < len(flows))

    def test_recall_with_and_without_twins(self):
        labels = pd.Series(["BENIGN", "BENIGN", "X", "X", "X", "X", "Y", "Z"])
        alerts = np.array([1, 0, 1, 1, 1, 0, 1, 0], dtype=np.int8)
        twins = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=bool)
        out = baseline.breakdown_by_attack(labels, alerts, twins, {"X": 9, "Y": 1})
        x, y = out["seen"]["X"], out["seen"]["Y"]
        self.assertEqual((x["n"], x["n_twins"], x["n_novel"]), (4, 2, 2))
        self.assertEqual(
            (x["recall"], x["recall_twins"], x["recall_novel"]), (0.75, 1.0, 0.5)
        )
        self.assertEqual(x["twin_inflation"], 0.25)
        # Sin flujos novedosos, el recall es indefinido y no entra al promedio.
        self.assertTrue(math.isnan(y["recall_novel"]))
        self.assertEqual(out["macro_recall_seen_novel"], 0.5)
        self.assertEqual(list(out["unseen"]), ["Z"])
        self.assertEqual((out["benign"]["fpr"], out["benign"]["fpr_novel"]), (0.5, 0.0))

    def test_everything_is_a_twin(self):
        splits = make_splits()
        splits["test"] = splits["train"].copy()
        _, report = baseline.fit_and_evaluate(splits, baseline.TrainParams(**FAST))
        self.assertIsNone(report["budgets"][0]["test_excluding_twins"])
        baseline.format_report(baseline.to_strict_json(report))

    def test_audit_in_report(self):
        splits = make_splits()
        _, report = baseline.fit_and_evaluate(splits, baseline.TrainParams(**FAST))
        audit = report["twin_audit"]["by_label"]
        self.assertEqual(audit["PortScan"]["twins"], 40)
        self.assertEqual(audit["SSH-Patator"]["twins"], 0)
        row = report["budgets"][0]
        self.assertEqual(row["by_label_raw"]["seen"]["PortScan"]["n_novel"], 110)
        self.assertEqual(
            row["test"]["n_attack"] - row["test_excluding_twins"]["n_attack"], 40
        )


class ParamsTests(unittest.TestCase):
    def test_scale_pos_weight(self):
        y = np.array([0, 0, 0, 1], dtype=np.int8)
        self.assertEqual(baseline.compute_scale_pos_weight(y, "train"), 3.0)
        self.assertEqual(baseline.compute_scale_pos_weight(y, "none"), 1.0)
        self.assertEqual(baseline.compute_scale_pos_weight(y, "2.5"), 2.5)
        with self.assertRaises(ValueError):
            baseline.compute_scale_pos_weight(np.zeros(3), "train")

    def test_weight_uses_only_train(self):
        splits = make_splits()
        rng = np.random.default_rng(9)
        splits["val"] = make_frame(rng, n_benign=200, PortScan=900)
        _, report = baseline.fit_and_evaluate(splits, baseline.TrainParams(**FAST))
        self.assertEqual(report["training"]["scale_pos_weight"], 1500 / 500)

    def test_invalid_params(self):
        for bad in ("otro", 0, -1, math.nan, math.inf, None):
            with self.subTest(balance=bad), self.assertRaises(ValueError):
                baseline.TrainParams(balance=bad)
        with self.assertRaises(ValueError):
            baseline.TrainParams(model="svm")
        with self.assertRaises(ValueError):
            baseline.TrainParams(device="tpu")

    def test_device_resolution(self):
        for available, expected in ((True, "cuda"), (False, "cpu")):
            with mock.patch.object(baseline, "cuda_available", return_value=available):
                self.assertEqual(baseline.resolve_device("auto", "xgb"), expected)
                self.assertEqual(baseline.resolve_device("cpu", "xgb"), "cpu")
                self.assertEqual(baseline.resolve_device("auto", "lgbm"), "cpu")
        with (
            mock.patch.object(baseline, "cuda_available", return_value=False),
            self.assertRaises(ValueError),
        ):
            baseline.resolve_device("cuda", "xgb")
        with self.assertRaises(ValueError):
            baseline.resolve_device("cuda", "lgbm")

    def test_cuda_detection(self):
        def detect(which, returncode, stdout, build=True):
            result = mock.Mock(returncode=returncode, stdout=stdout)
            with (
                mock.patch.object(baseline.shutil, "which", return_value=which),
                mock.patch.object(baseline.subprocess, "run", return_value=result),
                mock.patch.object(
                    baseline.xgb, "build_info", return_value={"USE_CUDA": build}
                ),
            ):
                return baseline.cuda_available()

        self.assertTrue(detect("nvidia-smi", 0, "GPU 0: Tesla T4"))
        self.assertFalse(detect(None, 0, "GPU 0"))
        self.assertFalse(detect("nvidia-smi", 9, ""))
        self.assertFalse(detect("nvidia-smi", 0, "No devices found"))
        self.assertFalse(detect("nvidia-smi", 0, "GPU 0", build=False))

    def test_val_needs_both_classes(self):
        splits = make_splits()
        val = splits["val"]
        splits["val"] = val[val[TARGET_COLUMN] == 0].reset_index(drop=True)
        with self.assertRaises(ValueError):
            baseline.fit_and_evaluate(splits, baseline.TrainParams(**FAST))

    def test_suffixes_never_collide(self):
        self.assertEqual(baseline.build_suffix(), "")
        self.assertEqual(baseline.build_suffix("v2"), "_v2")
        self.assertEqual(baseline.build_suffix("_v2"), "_v2")
        suffixes = {
            baseline.build_suffix("", model, drop)
            for model in baseline.MODELS
            for drop in (False, True)
        }
        self.assertEqual(len(suffixes), 4)
        for bad in ("../x", "a/b", "a b", "ñ"):
            with self.subTest(tag=bad), self.assertRaises(ValueError):
                baseline.build_suffix(bad)

    def test_default_paths_follow_config(self):
        self.assertEqual(
            config.get_baseline_model_path("cicids2017", "_v2"),
            config.MODELS_DIR / "xgb_baseline_cicids2017_v2.json",
        )
        self.assertEqual(
            config.get_baseline_model_path("cicids2017", "", "txt"),
            config.MODELS_DIR / "xgb_baseline_cicids2017.txt",
        )
        self.assertEqual(
            config.get_baseline_report_path("cicids2017"),
            config.METRICS_DIR / "xgb_baseline_cicids2017.json",
        )
        with self.assertRaises(ValueError):
            config.get_baseline_report_path("otro")


class DropPortTests(unittest.TestCase):
    def test_select_features(self):
        columns = [*FEATURES, *METADATA_COLUMNS]
        self.assertEqual(baseline.select_features(columns), (FEATURES, []))
        self.assertEqual(
            baseline.select_features(columns, drop_dst_port=True),
            (FEATURES[1:], [PORT]),
        )

    def test_missing_port_is_an_error(self):
        with self.assertRaises(ValueError):
            baseline.select_features(FEATURES[1:], drop_dst_port=True)

    def test_model_ignores_the_port(self):
        splits = make_splits()
        model, report = baseline.fit_and_evaluate(
            splits, baseline.TrainParams(**FAST), drop_dst_port=True
        )
        self.assertNotIn(PORT, model.features)
        self.assertNotIn(PORT, model.booster.feature_names)
        self.assertEqual(report["dropped_features"], [PORT])
        self.assertNotIn(
            PORT, [row["feature"] for row in report["top_features_by_gain"]]
        )
        changed = splits["test"].copy()
        changed[PORT] = np.float32(9999)
        np.testing.assert_array_equal(
            baseline.score_supervised(model, splits["test"]),
            baseline.score_supervised(model, changed),
        )


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.dirs = {
            name: root / name for name in ("processed_dir", "models_dir", "metrics_dir")
        }
        for path in self.dirs.values():
            path.mkdir()
        self.splits = make_splits()
        write_artifacts(self.dirs["processed_dir"], self.splits)

    def run_baseline(self, model="xgb", **kwargs):
        return baseline.run(
            params=baseline.TrainParams(model=model, **FAST), **self.dirs, **kwargs
        )

    def load(self, suffix=""):
        return baseline.load_baseline(
            suffix=suffix,
            models_dir=self.dirs["models_dir"],
            metrics_dir=self.dirs["metrics_dir"],
        )

    def files(self, name):
        return sorted(p.name for p in self.dirs[name].iterdir())

    def test_round_trip(self):
        test = self.splits["test"]
        for backend, suffix in (("xgb", ""), ("lgbm", "_lgbm")):
            with self.subTest(backend=backend):
                report = self.run_baseline(model=backend)
                expected, _ = baseline.fit_and_evaluate(
                    self.splits, baseline.TrainParams(model=backend, **FAST)
                )
                model, saved = self.load(suffix)
                self.assertEqual(saved, report)
                scores = baseline.score_supervised(model, test)
                np.testing.assert_array_equal(
                    scores, baseline.score_supervised(expected, test)
                )
                # El modelo recargado y el umbral guardado reproducen la matriz.
                row = saved["budgets"][0]
                hits = (scores >= row["threshold"]) & (test[TARGET_COLUMN] == 1)
                self.assertEqual(
                    int(hits.sum()), row["test"]["confusion_matrix"]["tp"]
                )
        self.assertEqual(
            self.files("models_dir"),
            ["xgb_baseline_cicids2017.json", "xgb_baseline_cicids2017_lgbm.txt"],
        )

    def test_reload_ignores_column_order(self):
        test = self.splits["test"]
        for backend, suffix in (("xgb", ""), ("lgbm", "_lgbm")):
            with self.subTest(backend=backend):
                self.run_baseline(model=backend)
                model, _ = self.load(suffix)
                np.testing.assert_array_equal(
                    baseline.score_supervised(model, test),
                    baseline.score_supervised(model, test[test.columns[::-1]]),
                )

    def test_lgbm_importance_keeps_original_names(self):
        report = self.run_baseline(model="lgbm")
        names = {row["feature"] for row in report["top_features_by_gain"]}
        self.assertIn("Flow Duration", names)
        self.assertLessEqual(names, set(FEATURES))

    def test_report_is_strict_json_without_nan(self):
        def reject(constant):
            raise AssertionError(f"{constant} en el reporte")

        self.run_baseline(run_ablation=True)
        for path in self.dirs["metrics_dir"].iterdir():
            text = path.read_text(encoding="utf-8")
            report = json.loads(text, parse_constant=reject)
            self.assertEqual(non_finite(report), [])
            self.assertEqual(report["score_dtype"], "float64")

    def test_undefined_metrics_become_null(self):
        value = {"a": math.nan, "b": [np.float64("inf"), np.int64(3)], 0.1: None}
        self.assertEqual(
            baseline.to_strict_json(value), {"a": None, "b": [None, 3], "0.1": None}
        )

    def test_ablation(self):
        report = self.run_baseline(run_ablation=True)
        ablation = report["ablation_dst_port"]
        self.assertEqual(ablation["dropped_features"], [PORT])
        self.assertEqual(ablation["max_fpr"], 0.001)
        _, without = self.load("_no_dst_port")
        metrics = without["budgets"][0]["test"]
        for key, value in (
            ("test_pr_auc", metrics["pr_auc"]),
            ("recall", metrics["recall"]),
        ):
            row = ablation["summary"][key]
            self.assertEqual(row["without_port"], value)
            self.assertAlmostEqual(row["delta"], value - row["with_port"])
        self.assertEqual(
            set(ablation["recall_by_label"]), {"PortScan", "SSH-Patator", "Heartbleed"}
        )
        # Con menos columnas, un gemelo lo sigue siendo: la fracción solo crece.
        for row in ablation["twin_fraction_by_label"].values():
            self.assertGreaterEqual(row["delta"], 0)
        own = ablation["excluding_own_twins"]
        self.assertEqual(
            own["without_port"]["n_attack"],
            without["budgets"][0]["test_excluding_twins"]["n_attack"],
        )
        self.assertEqual(len(self.files("models_dir")), 2)
        self.assertIn("Ablación", baseline.format_report(report))
        _, saved = self.load()
        self.assertEqual(saved, report)

    def test_ablation_fails_before_training(self):
        for name in SPLITS:
            self.splits[name] = self.splits[name].drop(columns=PORT)
        write_artifacts(self.dirs["processed_dir"], self.splits)
        with self.assertRaises(ValueError):
            self.run_baseline(run_ablation=True)
        self.assertEqual(self.files("models_dir"), [])

    def test_ablation_excludes_drop(self):
        with self.assertRaises(ValueError):
            self.run_baseline(run_ablation=True, drop_dst_port=True)

    def test_tag_does_not_overwrite(self):
        self.run_baseline()
        path = self.dirs["metrics_dir"] / "xgb_baseline_cicids2017.json"
        before = path.read_bytes()
        self.run_baseline(tag="v2", drop_dst_port=True)
        self.assertEqual(path.read_bytes(), before)
        self.assertIn(
            "xgb_baseline_cicids2017_v2_no_dst_port.json", self.files("models_dir")
        )

    def test_tampered_or_missing_model(self):
        with self.assertRaises(etl.ArtifactError):
            self.load()
        self.run_baseline()
        path = self.dirs["models_dir"] / "xgb_baseline_cicids2017.json"
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaises(etl.ArtifactError):
            self.load()
        path.unlink()
        with self.assertRaises(etl.ArtifactError):
            self.load()

    def test_other_features_are_rejected(self):
        for backend, name in (("xgb", ".json"), ("lgbm", "_lgbm.txt")):
            self.run_baseline(model=backend)
            path = self.dirs["models_dir"] / f"xgb_baseline_cicids2017{name}"
            with self.subTest(backend=backend), self.assertRaises(etl.ArtifactError):
                baseline.load_model(path, FEATURES[1:])


class CliTests(unittest.TestCase):
    def invoke(self, *argv):
        with (
            mock.patch.object(baseline, "run") as run,
            mock.patch.object(baseline, "ensure_directories"),
            mock.patch.object(baseline, "format_report", return_value=""),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            baseline.main(list(argv))
        return run.call_args.args

    def test_defaults(self):
        dataset, params, drop, ablation, tag = self.invoke()
        self.assertEqual(
            (dataset, drop, ablation, tag), ("cicids2017", False, False, "")
        )
        self.assertEqual(params, baseline.TrainParams())

    def test_flags(self):
        _, params, drop, _, tag = self.invoke(
            *("--model", "lgbm", "--n-estimators", "7", "--max-depth", "4"),
            *("--learning-rate", "0.2", "--balance", "3", "--device", "cpu"),
            *("--drop-dst-port", "--tag", "v2"),
        )
        self.assertEqual(
            params, baseline.TrainParams("lgbm", 7, 4, 0.2, balance=3.0, device="cpu")
        )
        self.assertEqual((drop, tag), (True, "v2"))

    def test_invalid(self):
        for argv in (
            ["--drop-dst-port", "--run-ablation"],
            ["--balance", "otro"],
            ["--n-estimators", "0"],
            ["--learning-rate", "-1"],
            ["--model", "svm"],
            ["--tag", "../x"],
            ["--model", "lgbm", "--device", "cuda"],
        ):
            with self.subTest(argv=argv), self.assertRaises(SystemExit) as caught:
                self.invoke(*argv)
            self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
