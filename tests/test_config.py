"""Pruebas rápidas de `src.config`."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src import config


def patched_paths(root: Path):
    """Redirige las carpetas de salida a `root`."""
    return mock.patch.multiple(
        config,
        RAW_DIR=root / "raw",
        PROCESSED_DIR=root / "processed",
        MODELS_DIR=root / "models",
        METRICS_DIR=root / "metrics",
        FIGURES_DIR=root / "figures",
    )


class StorageRootTest(unittest.TestCase):
    def test_local_when_drive_is_missing(self):
        root = config.resolve_storage_root(Path("/no/existe/MyDrive"))
        self.assertEqual(root, config.PROJECT_ROOT)

    def test_drive_when_mounted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = config.resolve_storage_root(Path(tmp))
            self.assertEqual(root, Path(tmp) / "hybrid_nids")

    def test_project_root_is_repo(self):
        self.assertTrue((config.PROJECT_ROOT / "requirements.txt").is_file())

    def test_paths_hang_from_storage_root(self):
        for path in (
            config.RAW_DIR,
            config.PROCESSED_DIR,
            config.MODELS_DIR,
            config.METRICS_DIR,
            config.FIGURES_DIR,
        ):
            self.assertTrue(path.is_relative_to(config.STORAGE_ROOT))


class ConstantsTest(unittest.TestCase):
    def test_globals(self):
        self.assertEqual(config.SUPPORTED_DATASETS, ("cicids2017", "unsw_nb15"))
        self.assertEqual(config.DEFAULT_DATASET, "cicids2017")
        self.assertEqual(config.RANDOM_SEED, 42)
        self.assertEqual(config.TARGET_FPR_BUDGETS, (0.001, 0.005, 0.01))


class PathHelpersTest(unittest.TestCase):
    def test_split_path(self):
        path = config.get_split_path("unsw_nb15", "val")
        self.assertEqual(path, config.PROCESSED_DIR / "unsw_nb15" / "val.parquet")

    def test_rejects_unknown_dataset_and_split(self):
        with self.assertRaises(ValueError):
            config.get_raw_dir("kdd99")
        with self.assertRaises(ValueError):
            config.get_split_path("cicids2017", "holdout")

    def test_ensure_directories_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patched_paths(root):
                config.ensure_directories()
                config.ensure_directories()
            self.assertTrue((root / "raw" / "cicids2017").is_dir())
            self.assertTrue((root / "processed" / "unsw_nb15").is_dir())
            self.assertTrue((root / "figures").is_dir())

    def test_warns_on_colab_without_drive(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patched_paths(Path(tmp)),
            mock.patch.dict("os.environ", {"COLAB_RELEASE_TAG": "x"}),
            mock.patch.object(config, "IN_COLAB", False),
            self.assertWarns(UserWarning),
        ):
            config.ensure_directories()


class LeakageTest(unittest.TestCase):
    def test_detects_cic_headers_with_spaces(self):
        columns = ["Flow ID", " Source IP", " Timestamp", " Flow Duration"]
        self.assertEqual(config.find_leakage_columns(columns), columns[:3])

    def test_detects_unsw_columns(self):
        columns = ["srcip", "sport", "dstip", "Stime", "Ltime", "id", "dur"]
        self.assertEqual(config.find_leakage_columns(columns), columns[:-1])

    def test_keeps_behavioral_features(self):
        columns = ["Flow Duration", "Idle Mean", "Fwd IAT Mean", "ct_src_dport_ltm"]
        self.assertEqual(config.find_leakage_columns(columns), [])

    def test_shortcut_columns_are_not_pruned(self):
        shortcuts = config.SHORTCUT_PRONE_COLUMNS
        self.assertEqual(config.find_leakage_columns(shortcuts), [])


if __name__ == "__main__":
    unittest.main()
