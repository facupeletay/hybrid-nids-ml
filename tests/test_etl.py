"""Pruebas de `src.etl` con CSV sintéticos que replican CIC-IDS2017."""

import functools
import http.server
import io
import json
import sys
import tempfile
import threading
import types
import unittest
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from src import etl
from src.config import (
    LEAKAGE_COLUMNS,
    METADATA_COLUMNS,
    SPLITS,
    find_leakage_columns,
    get_feature_columns,
)

# Encabezado con los espacios y la columna repetida del original.
HEADER = [
    "Flow ID", " Source IP", " Source Port", " Destination IP",
    " Destination Port", " Protocol", " Timestamp", " Flow Duration",
    "Flow Bytes/s", " Flow Packets/s", " Flow IAT Min", " Fwd Header Length",
    " Fwd Header Length.1", " Bwd PSH Flags", "Init_Win_bytes_forward",
    " Label",
]  # fmt: skip

FILE_NAMES = {
    "monday": "Monday-WorkingHours.pcap_ISCX.csv",
    "tuesday": "Tuesday-WorkingHours.pcap_ISCX.csv",
    "wednesday": "Wednesday-workingHours.pcap_ISCX.csv",
    "web": "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "infiltration": "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "bot": "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "portscan": "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "ddos": "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
}

N_CAMPAIGN = 100
CONFIG = etl.EtlConfig(min_campaign_flows=20)


def clock(moment, seconds=False):
    """Formato de CIC-IDS2017: d/m/aaaa y reloj de 12 h sin AM/PM."""
    hour = (moment.hour - 1) % 12 + 1
    if seconds:
        return f"{moment:%d/%m/%Y} {hour:02d}:{moment:%M:%S}"
    return f"{moment.day}/{moment.month}/{moment.year} {hour}:{moment:%M}"


class FlowFactory:
    """Genera filas con la forma del CSV original, con puertos únicos."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.port = 1024

    def row(self, day, start, label, seconds=False, port=None, **values):
        if port is None:
            self.port += 1
            port = self.port
        moment = datetime(2017, 7, day) + start
        row = {
            "Flow ID": f"10.0.0.1-10.0.0.2-{port}-80-6",
            " Source IP": "10.0.0.1",
            " Source Port": port,
            " Destination IP": "10.0.0.2",
            " Destination Port": 80,
            " Protocol": 6,
            " Timestamp": clock(moment, seconds),
            " Flow Duration": int(self.rng.integers(1, 1_000_000)),
            "Flow Bytes/s": float(self.rng.random() * 1e4),
            " Flow Packets/s": float(self.rng.random() * 1e3),
            " Flow IAT Min": int(self.rng.integers(0, 100)),
            " Fwd Header Length": int(self.rng.integers(20, 200)),
            " Bwd PSH Flags": 0,
            "Init_Win_bytes_forward": -1 if self.rng.random() < 0.5 else 8192,
            " Label": label,
        }
        row.update(values)
        row[" Fwd Header Length.1"] = row[" Fwd Header Length"]
        return row

    def campaign(self, day, hour, label, n=N_CAMPAIGN, seconds=False):
        """`n` flujos, uno por minuto, desde `hour`."""
        return [
            self.row(day, timedelta(hours=hour, minutes=i), label, seconds)
            for i in range(n)
        ]


def build_rows():
    """Filas por archivo, con todas las trampas."""
    make = FlowFactory()
    benign = functools.partial(make.campaign, label="BENIGN")
    rows = {
        "monday": benign(3, 9, seconds=True) + benign(3, 13, seconds=True),
        "tuesday": benign(4, 9)
        + make.campaign(4, 10, "FTP-Patator")
        + make.campaign(4, 14, "SSH-Patator"),
        "wednesday": benign(5, 12)  # cruza el mediodía: 12:59 -> 1:00
        + make.campaign(5, 9, "DoS Hulk")
        + make.campaign(5, 15, "Heartbleed", n=5),
        "web": benign(6, 9)
        + make.campaign(6, 10, "Web Attack \x96 Brute Force")
        + make.campaign(6, 11, "Web Attack \x96 XSS", n=25)
        + make.campaign(6, 10, "Web Attack \x96 Sql Injection", n=3),
        "infiltration": benign(6, 13) + make.campaign(6, 14, "Infiltration", n=6),
        "bot": benign(7, 9) + make.campaign(7, 10, "Bot"),
        "portscan": benign(7, 13) + make.campaign(7, 13, "PortScan"),
        "ddos": benign(7, 15) + make.campaign(7, 15, "DDoS"),
    }
    at = timedelta
    rows["tuesday"] += [
        # Duración 0: tasas indefinidas.
        make.row(4, at(hours=9), "BENIGN", **{
            " Flow Duration": 0, "Flow Bytes/s": "Infinity",
            " Flow Packets/s": "Infinity"}),
        make.row(4, at(hours=9), "BENIGN", **{
            " Flow Duration": 0, "Flow Bytes/s": "NaN",
            " Flow Packets/s": "-Infinity"}),
        # Flujos corruptos.
        make.row(4, at(hours=9), "BENIGN", **{" Flow Duration": -1}),
        make.row(4, at(hours=9), "BENIGN", **{" Fwd Header Length": -32212234632}),
        # Reordenamiento de paquetes: se conserva con IAT 0.
        make.row(4, at(hours=9), "FTP-Patator", **{" Flow IAT Min": -12}),
    ]  # fmt: skip
    # Duplicado exacto, metadata incluida.
    rows["monday"].append(dict(rows["monday"][0]))
    # Una sesión TCP partida por el timeout, a ambos lados del corte Train/Val.
    for minute in (68, 69, 70, 71):
        start = timedelta(hours=9, minutes=minute)
        rows["bot"].append(make.row(7, start, "BENIGN", port=60000))
    # Mismo puerto, pero lejos en el tiempo: reuso legítimo, no se purga.
    for minute in (0, 95):
        start = timedelta(hours=9, minutes=minute)
        rows["bot"].append(make.row(7, start, "BENIGN", port=60001))
    return rows


def write_csvs(rows, folder, empty_rows=50):
    """Escribe los CSV en cp1252, desordenados y con filas vacías."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    for key, file_rows in rows.items():
        frame = pd.DataFrame(file_rows, columns=HEADER)
        frame = frame.iloc[rng.permutation(len(frame))]
        text = frame.to_csv(index=False, lineterminator="\n")
        if key == "web":
            text += ("," * (len(HEADER) - 1) + "\n") * empty_rows
        (folder / FILE_NAMES[key]).write_bytes(text.encode("latin-1"))


class EtlTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def run_etl(self, rows, name="run", config=CONFIG):
        raw, out = self.root / name / "raw", self.root / name / "out"
        write_csvs(rows, raw)
        etl.run_etl("cicids2017", raw_dir=raw, output_dir=out, config=config)
        return out


class PipelineTest(unittest.TestCase):
    """Verificaciones de solo lectura sobre una única corrida."""

    @classmethod
    def setUpClass(cls):
        tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.raw, cls.out = Path(tmp.name) / "raw", Path(tmp.name) / "out"
        write_csvs(build_rows(), cls.raw)
        etl.run_etl("cicids2017", raw_dir=cls.raw, output_dir=cls.out, config=CONFIG)
        cls.artifacts = etl.load_artifacts("cicids2017", output_dir=cls.out)
        cls.metadata = cls.artifacts.metadata
        flows, _ = etl.load_raw("cicids2017", cls.raw)
        cls.frames, _ = etl.split_flows(flows, CONFIG)

    def source(self, campaign):
        sources = self.metadata["sources"]
        return next(s for s in sources if s["campaign_day"] == campaign)

    # ------------------------------------------------------------- artefactos
    def test_expected_files(self):
        self.assertEqual(
            sorted(p.name for p in self.out.iterdir()),
            [
                "etl_metadata_cicids2017.json",
                "scaler_cicids2017.joblib",
                "test_cicids2017.parquet",
                "train_cicids2017.parquet",
                "val_cicids2017.parquet",
            ],
        )

    def test_no_forbidden_columns(self):
        for name in SPLITS:
            columns = pd.read_parquet(self.out / f"{name}_cicids2017.parquet").columns
            self.assertEqual(find_leakage_columns(columns), [])
            for forbidden in (*LEAKAGE_COLUMNS, *etl.INTERNAL_COLUMNS):
                self.assertNotIn(forbidden, columns)

    def test_same_schema_in_every_split(self):
        for frame in self.artifacts.splits.values():
            self.assertEqual(
                list(frame.columns),
                [*self.artifacts.feature_columns, *METADATA_COLUMNS],
            )
            features = frame[get_feature_columns(frame.columns)]
            self.assertTrue((features.dtypes == np.float32).all())
            self.assertTrue(np.isfinite(features.to_numpy()).all())

    def test_pruned_columns(self):
        columns = self.metadata["columns"]
        self.assertEqual(
            sorted(columns["dropped_leakage"]),
            ["Destination IP", "Flow ID", "Source IP", "Source Port", "Timestamp"],
        )
        self.assertEqual(
            sorted(columns["dropped_constant_in_train"]),
            ["Bwd PSH Flags", "Destination Port", "Protocol"],
        )
        for name in ("Bwd PSH Flags", "Fwd Header Length.1"):
            self.assertNotIn(name, columns["features"])
        self.assertIn("Fwd Header Length", columns["features"])

    # --------------------------------------------------------------- limpieza
    def test_empty_rows(self):
        web = self.source("thursday_morning_webattacks")
        self.assertEqual(web["empty_rows_dropped"], 50)

    def test_repeated_column(self):
        for source in self.metadata["sources"]:
            self.assertEqual(
                source["repeated_columns_dropped"], ["Fwd Header Length.1"]
            )

    def test_invalid_rows(self):
        tuesday = self.source("tuesday")
        self.assertEqual(tuesday["nan_or_inf_rows_dropped"], {"BENIGN": 2})
        self.assertEqual(tuesday["negative_rows_dropped"], {"BENIGN": 2})
        self.assertEqual(
            tuesday["infinite_values"], {"Flow Bytes/s": 1, "Flow Packets/s": 2}
        )

    def test_negative_iat_is_clipped_not_dropped(self):
        self.assertEqual(self.source("tuesday")["negative_iat_rows"], {"FTP-Patator": 1})
        for frame in self.artifacts.splits.values():
            self.assertGreaterEqual(frame["Flow IAT Min"].min(), 0)
        total = sum(
            c["by_label"].get("FTP-Patator", 0)
            for c in self.metadata["counts"].values()
        )
        self.assertEqual(total, N_CAMPAIGN + 1)

    def test_sentinel_is_kept(self):
        train = self.artifacts.splits["train"]
        self.assertTrue((train["Init_Win_bytes_forward"] == -1).any())

    def test_exact_duplicate(self):
        monday = self.source("monday")
        self.assertEqual(monday["exact_duplicates_dropped"], {"BENIGN": 1})
        self.assertEqual(monday["rows_kept"], 2 * N_CAMPAIGN)

    # -------------------------------------------------------------- etiquetas
    def test_labels_and_families(self):
        full = pd.concat(self.artifacts.splits.values())
        pairs = dict(zip(full["label_raw"], full["attack_family"]))
        self.assertEqual(
            pairs,
            {
                "BENIGN": "BENIGN",
                "FTP-Patator": "BruteForce",
                "SSH-Patator": "BruteForce",
                "DoS Hulk": "DoS",
                "Heartbleed": "DoS",
                "Web Attack - Brute Force": "WebAttack",
                "Web Attack - XSS": "WebAttack",
                "Web Attack - Sql Injection": "WebAttack",
                "Infiltration": "Infiltration",
                "Bot": "Bot",
                "PortScan": "PortScan",
                "DDoS": "DDoS",
            },
        )
        benign = full["attack_family"] == "BENIGN"
        self.assertTrue((full["is_attack"] == (~benign).astype(int)).all())

    # ------------------------------------------------------------------ split
    def test_strict_chronological_order(self):
        tagged = pd.concat(
            frame.assign(split=code)
            for code, frame in enumerate(self.frames.values())
        )
        checked = 0
        for _, group in tagged.groupby(["campaign_day", "label_raw"]):
            first = group.groupby("split")[etl.START_COLUMN].min()
            last = group.groupby("split")[etl.START_COLUMN].max()
            for early in last.index:
                for late in first.index[first.index > early]:
                    self.assertLess(last[early], first[late])
                    checked += 1
        self.assertGreaterEqual(checked, 30)

    def test_fractions(self):
        rule = next(
            r
            for r in self.metadata["split"]["campaigns"]
            if r["label_raw"] == "PortScan"
        )
        self.assertEqual(
            [rule[f"{s}_flows"] for s in SPLITS],
            [0.70 * N_CAMPAIGN, 0.15 * N_CAMPAIGN, 0.15 * N_CAMPAIGN],
        )

    def test_small_campaigns_go_to_test(self):
        small = {
            r["label_raw"]
            for r in self.metadata["split"]["campaigns"]
            if r["rule"] == "test_only_small_campaign"
        }
        expected = {"Heartbleed", "Infiltration", "Web Attack - Sql Injection"}
        self.assertEqual(small, expected)
        for name in ("train", "val"):
            labels = set(self.artifacts.splits[name]["label_raw"])
            self.assertFalse(labels & expected)

    def test_no_session_across_boundary(self):
        full = pd.concat(
            frame.assign(split=code)
            for code, frame in enumerate(self.frames.values())
        )
        pairs = full.merge(full, on=etl.FLOW_KEY_COLUMN)
        pairs = pairs[pairs["split_x"] < pairs["split_y"]]
        gap = np.maximum(
            pairs["_start_y"] - pairs["_end_x"], pairs["_start_x"] - pairs["_end_y"]
        )
        self.assertTrue((gap > CONFIG.session_gap_seconds).all())
        # El reuso lejano del puerto 60001 sobrevive en dos splits.
        self.assertGreaterEqual(len(pairs), 1)

    def test_purge_hits_earlier_side_only(self):
        purge = self.metadata["split"]["session_purge"]
        self.assertGreaterEqual(purge["train"].get("BENIGN", 0), 1)
        self.assertEqual(purge["test"], {})

    # ----------------------------------------------------------------- scaler
    def test_scaler_is_fitted_on_benign_train(self):
        train = self.artifacts.splits["train"]
        benign = train.loc[train["is_attack"] == 0, self.artifacts.feature_columns]
        scaler = self.artifacts.scaler
        np.testing.assert_allclose(
            scaler.mean_, benign.astype(np.float64).mean(), rtol=1e-6
        )
        self.assertEqual(self.metadata["scaler"]["flows"], len(benign))

    def test_metadata_is_strict_json_and_traceable(self):
        text = (self.out / "etl_metadata_cicids2017.json").read_text("utf-8")
        data = json.loads(text, parse_constant=self.fail)
        self.assertEqual(data["schema_version"], etl.SCHEMA_VERSION)
        self.assertEqual(len(data["sources"]), 8)
        self.assertEqual(len(data["config_hash"]), 64)
        for source in data["sources"]:
            self.assertEqual(len(source["sha256"]), 64)
        for name, frame in self.artifacts.splits.items():
            self.assertEqual(data["counts"][name]["total"], len(frame))


class IsolationTest(EtlTestCase):
    """Alterar Val/Test no puede cambiar lo que se aprende de Train."""

    def test_val_and_test_do_not_influence_train(self):
        base = self.run_etl(build_rows(), "base")
        reference = etl.load_artifacts("cicids2017", output_dir=base)
        rules = reference.metadata["split"]["campaigns"]
        cuts = {
            (r["campaign_day"], r["label_raw"]): r["val_start"]
            for r in rules
            if r["rule"] == "chronological"
        }

        adapter = etl.ADAPTERS["cicids2017"]
        rows, altered = build_rows(), 0
        for key, file_rows in rows.items():
            campaign, day = adapter.files[adapter._file_key(FILE_NAMES[key])]
            for row in file_rows:
                label = adapter.normalize_label(row[" Label"])
                stamp = adapter.parse_timestamp(row[" Timestamp"], day)
                cut = cuts.get((campaign, label))
                if cut is not None and stamp < datetime.fromisoformat(cut):
                    continue
                # Valores extremos, y varianza donde Train es constante. La
                # 5-tupla no se toca: define la sesión, no es una feature más.
                row["Flow Bytes/s"] = 9e8 + altered
                row[" Fwd Header Length"] = 60000
                row[" Fwd Header Length.1"] = 7
                row[" Bwd PSH Flags"] = altered % 2
                altered += 1
        self.assertGreater(altered, 100)

        other = etl.load_artifacts(
            "cicids2017", output_dir=self.run_etl(rows, "altered")
        )
        for key in (
            "features",
            "dropped_constant_in_train",
            "dropped_leakage",
        ):
            self.assertEqual(
                reference.metadata["columns"][key], other.metadata["columns"][key]
            )
        pd.testing.assert_frame_equal(
            reference.splits["train"], other.splits["train"]
        )
        np.testing.assert_array_equal(reference.scaler.mean_, other.scaler.mean_)
        np.testing.assert_array_equal(reference.scaler.scale_, other.scaler.scale_)
        self.assertFalse(
            reference.splits["test"].equals(other.splits["test"]),
            "la alteración no llegó a Test: la prueba no probaría nada",
        )

    def test_attacks_in_train_do_not_influence_scaler(self):
        rows = build_rows()
        base = self.run_etl(rows, "base")
        for file_rows in rows.values():
            for row in file_rows:
                if row[" Label"] != "BENIGN":
                    row["Flow Bytes/s"] = 5e8
        other = self.run_etl(rows, "altered")
        a, b = (etl.load_artifacts("cicids2017", output_dir=o) for o in (base, other))
        np.testing.assert_array_equal(a.scaler.mean_, b.scaler.mean_)
        np.testing.assert_array_equal(a.scaler.scale_, b.scaler.scale_)

    def test_deterministic(self):
        outs = [self.run_etl(build_rows(), name) for name in ("a", "b")]
        hashes = [
            etl.load_artifacts("cicids2017", output_dir=o).metadata["artifacts"]
            for o in outs
        ]
        for name in SPLITS:
            self.assertEqual(hashes[0][name]["sha256"], hashes[1][name]["sha256"])


class SplitUnitTest(unittest.TestCase):
    def flows(self, items):
        """items: (clave, inicio, fin, split)."""
        key, start, end, split = zip(*items)
        frame = pd.DataFrame(
            {
                etl.FLOW_KEY_COLUMN: np.array(key, dtype=np.uint64),
                etl.START_COLUMN: np.array(start, dtype=float),
                etl.END_COLUMN: np.array(end, dtype=float),
            }
        )
        return frame, np.array(split, dtype=np.int8)

    def test_boundary_flows(self):
        frame, split = self.flows(
            [
                (1, 0, 10, 0),  # lejos del flujo de Test: se conserva
                (1, 500, 590, 0),  # a 10 s: se purga
                (1, 600, 610, 2),
                (2, 500, 590, 0),  # otra 5-tupla
                (3, 100, 900, 0),  # largo, lo contiene: se purga
                (3, 300, 310, 1),
                (4, 620, 630, 0),  # posterior a Test (otra campaña): se purga
                (4, 600, 610, 2),
                (5, 400, 500, 1),  # Val pegado a Test: se purga
                (5, 520, 530, 2),
                (5, 330, 350, 0),  # ...y Train queda lejos de lo que sobrevive
            ]
        )
        purge = etl.find_boundary_flows(frame, split, 60)
        self.assertEqual(
            purge.tolist(),
            [False, True, False, False, True, False, True, False, True, False, False],
        )

    def test_port_reuse_chain_does_not_empty_train(self):
        # Un flujo por minuto con la misma 5-tupla, como DoS Hulk.
        items = [(1, 60 * i, 60 * i + 30, 0 if i < 70 else 2) for i in range(100)]
        purge = etl.find_boundary_flows(*self.flows(items), 65)
        self.assertEqual(np.flatnonzero(purge).tolist(), [69])

    def test_cut_points_never_split_ties(self):
        start = pd.Series([0.0] * 90 + [60.0] * 5 + [120.0] * 5)
        self.assertEqual(etl.find_cut_points(start, etl.EtlConfig()), (60.0, 120.0))

    def test_too_few_timestamps(self):
        start = pd.Series([0.0] * 50 + [60.0] * 50)
        self.assertIsNone(etl.find_cut_points(start, etl.EtlConfig()))

    def test_config_is_validated(self):
        with self.assertRaises(ValueError):
            etl.EtlConfig(train_fraction=0.9, val_fraction=0.1)


class ParsingTest(unittest.TestCase):
    adapter = etl.ADAPTERS["cicids2017"]

    def parse(self, text, day):
        return self.adapter.parse_timestamp(text, date(2017, 7, day))

    def test_timestamps(self):
        cases = [
            ("5/7/2017 8:42", 5, (8, 42, 0)),
            ("5/7/2017 12:59", 5, (12, 59, 0)),
            ("5/7/2017 1:00", 5, (13, 0, 0)),
            ("6/7/2017 5:04", 6, (17, 4, 0)),
            ("03/07/2017 03:24:07", 3, (15, 24, 7)),
            ("7/6/2017 3:15", 6, (15, 15, 0)),  # mes primero
            ("2017-07-06 15:15:00", 6, (15, 15, 0)),  # ISO, 24 h
            ("7/7/2017 3:30 PM", 7, (15, 30, 0)),
            ("7/7/2017 12:05 PM", 7, (12, 5, 0)),
        ]
        for text, day, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(
                    self.parse(text, day), datetime(2017, 7, day, *expected)
                )

    def test_rejects_bad_timestamps(self):
        for text in ("", "ayer", "8/7/2017 9:00", "5/7/2017 23:00", "nan"):
            with self.subTest(text=text), self.assertRaises(etl.RawDataError):
                self.parse(text, 5)

    def test_web_attack_variants(self):
        for dash in ("\x96", "–", "—", "-", "�", "ï¿½", ""):
            label = f"Web Attack {dash} XSS"
            with self.subTest(label=label):
                self.assertEqual(
                    self.adapter.normalize_label(label), "Web Attack - XSS"
                )

    def test_label_case_and_spaces(self):
        self.assertEqual(self.adapter.normalize_label(" dos  Slowloris "), "DoS slowloris")

    def test_unknown_label(self):
        with self.assertRaises(etl.RawDataError):
            self.adapter.normalize_label("Ransomware")

    def test_columns_equal_after_strip(self):
        frame = pd.DataFrame([[1, 2, 3]], columns=["A", " A", "B.1"])
        out, dropped = self.adapter.normalize_columns(frame)
        self.assertEqual(list(out.columns), ["A", "B.1"])
        self.assertEqual(dropped, ["A"])

    def test_flow_key_is_bidirectional(self):
        frame = pd.DataFrame(
            {
                "Source IP": ["10.0.0.1", "10.0.0.2", "10.0.0.1"],
                "Source Port": [5000, 80, 5001],
                "Destination IP": ["10.0.0.2", "10.0.0.1", "10.0.0.2"],
                "Destination Port": [80, 5000, 80],
                "Protocol": [6, 6, 6],
            }
        )
        keys = self.adapter._flow_keys(frame, date(2017, 7, 3))
        self.assertEqual(keys[0], keys[1])
        self.assertNotEqual(keys[0], keys[2])


class SourcesTest(EtlTestCase):
    adapter = etl.ADAPTERS["cicids2017"]

    def test_machine_learning_csv_is_rejected(self):
        rows = build_rows()
        write_csvs(rows, self.root)
        for path in self.root.glob("*.csv"):
            frame = pd.read_csv(path, encoding="latin-1")
            drop = ["Flow ID", " Source IP", " Source Port", " Destination IP",
                    " Protocol", " Timestamp"]  # fmt: skip
            frame.drop(columns=drop).to_csv(path, index=False)
        with self.assertRaisesRegex(etl.RawDataError, "MachineLearningCSV"):
            self.adapter.find_sources(self.root)

    def test_prefers_labelled_flows_when_both_are_present(self):
        write_csvs(build_rows(), self.root / "TrafficLabelling")
        poor = self.root / "MachineLearningCVE"
        poor.mkdir()
        for name in FILE_NAMES.values():
            (poor / name).write_text(" Destination Port, Label\n80,BENIGN\n")
        sources = self.adapter.find_sources(self.root)
        self.assertEqual(len(sources), 8)
        self.assertEqual({p.parent.name for p in sources}, {"TrafficLabelling"})

    def test_missing_files(self):
        write_csvs(build_rows(), self.root)
        (self.root / FILE_NAMES["ddos"]).unlink()
        with self.assertRaisesRegex(etl.RawDataError, "ddos"):
            self.adapter.find_sources(self.root)

    def test_utf8_bom_is_accepted(self):
        write_csvs(build_rows(), self.root)
        path = self.root / FILE_NAMES["monday"]
        path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
        frame, _ = self.adapter.load(path)
        self.assertIn("Flow ID", frame.columns)
        self.assertEqual(len(self.adapter.find_sources(self.root)), 8)

    def test_kaggle_mirror(self):
        mirror, raw = self.root / "mirror", self.root / "raw"
        write_csvs(build_rows(), mirror / "TrafficLabelling")
        fake = types.SimpleNamespace(dataset_download=mock.Mock(return_value=str(mirror)))
        with mock.patch.dict(sys.modules, {"kagglehub": fake}):
            self.adapter.obtain(raw, "user/dataset")
            self.adapter.obtain(raw, "user/dataset")  # ya están: no descarga
        fake.dataset_download.assert_called_once_with("user/dataset")
        self.assertEqual(
            sorted(p.name for p in raw.iterdir()), sorted(FILE_NAMES.values())
        )

    def test_unsw_is_pending(self):
        with self.assertRaises(NotImplementedError):
            etl.run_etl("unsw_nb15", raw_dir=self.root, output_dir=self.root)


class LocalServer:
    """Servidor HTTP mínimo con soporte de `Range`."""

    def __init__(self, body, ranges=True, truncate=None):
        self.requests = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                header = self.headers.get("Range")
                outer.requests.append(header)
                start = 0
                if header and ranges:
                    start = int(header.removeprefix("bytes=").rstrip("-"))
                    if start >= len(body):
                        self.send_error(416)
                        return
                payload = body[start:]
                self.send_response(206 if header and ranges else 200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload if truncate is None else payload[:truncate])

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/flows.zip"

    def __enter__(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


class DownloadTest(EtlTestCase):
    adapter = etl.ADAPTERS["cicids2017"]

    def zip_bytes(self):
        write_csvs(build_rows(), self.root / "src")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as bundle:
            for path in (self.root / "src").iterdir():
                bundle.write(path, f"TrafficLabelling /{path.name}")
            bundle.writestr("../../evil.txt", "x")
        return buffer.getvalue()

    def obtain(self, server):
        raw = self.root / "raw"
        with mock.patch.object(self.adapter, "url", server.url):
            self.adapter.obtain(raw)
        return raw

    def test_download_extract_and_skip(self):
        with LocalServer(self.zip_bytes()) as server:
            raw = self.obtain(server)
            self.obtain(server)
            self.assertEqual(len(server.requests), 1)
        self.assertEqual(
            sorted(p.name for p in raw.iterdir()), sorted(FILE_NAMES.values())
        )
        self.assertFalse(list(self.root.rglob("evil.txt")))

    def test_html_instead_of_zip(self):
        with LocalServer(b"<!DOCTYPE HTML><html>registro</html>") as server:
            with self.assertRaisesRegex(etl.RawDataError, "formulario"):
                self.obtain(server)
        self.assertEqual(list((self.root / "raw").iterdir()), [])

    def test_user_zip_is_kept(self):
        raw = self.root / "raw"
        raw.mkdir()
        archive = raw / "GeneratedLabelledFlows.zip"
        archive.write_bytes(self.zip_bytes())
        self.adapter.obtain(raw)
        self.assertTrue(archive.exists())
        self.assertEqual(len(self.adapter.find_sources(raw)), 8)

    def test_resume(self):
        body = bytes(range(256)) * 100
        target = self.root / "f.bin"
        (self.root / "f.bin.part").write_bytes(body[:10_000])
        with LocalServer(body) as server:
            etl.download_file(server.url, target)
            self.assertEqual(server.requests, ["bytes=10000-"])
        self.assertEqual(target.read_bytes(), body)

    def test_server_without_range_restarts(self):
        body = b"abc" * 1000
        target = self.root / "f.bin"
        (self.root / "f.bin.part").write_bytes(b"basura")
        with LocalServer(body, ranges=False) as server:
            etl.download_file(server.url, target)
        self.assertEqual(target.read_bytes(), body)

    def test_complete_part_is_promoted(self):
        body = b"abc" * 1000
        target = self.root / "f.bin"
        (self.root / "f.bin.part").write_bytes(body)
        with LocalServer(body) as server:
            etl.download_file(server.url, target)
        self.assertEqual(target.read_bytes(), body)

    def test_truncated_download_keeps_part(self):
        with LocalServer(b"x" * 5000, truncate=1000) as server:
            with self.assertRaises(etl.RawDataError):
                etl.download_file(server.url, self.root / "f.bin")
        self.assertFalse((self.root / "f.bin").exists())
        self.assertTrue((self.root / "f.bin.part").exists())


class LoadArtifactsTest(EtlTestCase):
    def load(self, out):
        return etl.load_artifacts("cicids2017", output_dir=out)

    def test_missing(self):
        with self.assertRaisesRegex(etl.ArtifactError, "src.etl"):
            self.load(self.root)

    def test_tampered_parquet_and_scaler(self):
        out = self.run_etl(build_rows())
        for name in ("val_cicids2017.parquet", "scaler_cicids2017.joblib"):
            path = out / name
            original = path.read_bytes()
            path.write_bytes(original + b"\0")
            with self.subTest(name=name), self.assertRaises(etl.ArtifactError):
                self.load(out)
            path.write_bytes(original)
        self.load(out)

    def test_other_schema_version(self):
        out = self.run_etl(build_rows())
        with mock.patch.object(etl, "SCHEMA_VERSION", "99"):
            with self.assertRaisesRegex(etl.ArtifactError, "Esquema"):
                self.load(out)

    def test_partial_load(self):
        out = self.run_etl(build_rows())
        loaded = etl.load_artifacts("cicids2017", output_dir=out, splits=["val"])
        self.assertEqual(list(loaded.splits), ["val"])

    def test_failed_write_leaves_previous_file(self):
        path = self.root / "a.txt"
        path.write_text("previo")

        def broken(tmp):
            tmp.write_text("a medias")
            raise OSError("disco lleno")

        with self.assertRaises(OSError):
            etl._atomic_write(path, broken)
        self.assertEqual(path.read_text(), "previo")
        self.assertEqual([p.name for p in self.root.iterdir()], ["a.txt"])


if __name__ == "__main__":
    unittest.main()
