"""Pruebas rápidas de `src.metrics`."""

import json
import math
import unittest

import numpy as np

from src.config import TARGET_FPR_BUDGETS
from src.metrics import (
    compute_binary_soc_metrics,
    compute_per_attack_recall,
    find_threshold_for_fpr,
)


class BinarySocMetricsTest(unittest.TestCase):
    def test_hand_computed_case(self):
        y_true = [0, 0, 0, 0, 1, 1, 1, 1]
        y_score = [0.1, 0.2, 0.3, 0.9, 0.4, 0.6, 0.8, 0.95]
        m = compute_binary_soc_metrics(y_true, y_score, threshold=0.5)
        self.assertEqual(m["confusion_matrix"], {"tn": 3, "fp": 1, "fn": 1, "tp": 3})
        self.assertAlmostEqual(m["precision"], 0.75)
        self.assertAlmostEqual(m["recall"], 0.75)
        self.assertAlmostEqual(m["fpr"], 0.25)
        self.assertAlmostEqual(m["f1"], 0.75)
        # 13 de 16 pares (ataque, benigno) bien ordenados.
        self.assertAlmostEqual(m["roc_auc"], 13 / 16)
        # Precisión en cada acierto: 1/1, 2/3, 3/4, 4/5.
        self.assertAlmostEqual(m["pr_auc"], (1 + 2 / 3 + 3 / 4 + 4 / 5) / 4)
        self.assertAlmostEqual(m["alerts_per_100k_benign"], 25_000)
        json.dumps(m)

    def test_threshold_is_inclusive(self):
        m = compute_binary_soc_metrics([0, 1], [0.1, 0.5], threshold=0.5)
        self.assertEqual(m["recall"], 1.0)

    def test_undefined_metrics_are_nan(self):
        m = compute_binary_soc_metrics([0, 0], [0.1, 0.2], threshold=0.5)
        for key in ("recall", "roc_auc", "pr_auc", "precision"):
            self.assertTrue(math.isnan(m[key]), key)
        self.assertEqual(m["fpr"], 0.0)

    def test_rejects_bad_input(self):
        cases = [
            ([0, 2], [0.1, 0.2]),
            ([0, 1], [0.1, np.nan]),
            ([0, 1], [0.1, np.inf]),
            ([0, 1], [0.1]),
            ([], []),
        ]
        for y_true, y_score in cases:
            with self.subTest(y_true=y_true), self.assertRaises(ValueError):
                compute_binary_soc_metrics(y_true, y_score, threshold=0.5)


class ThresholdForFprTest(unittest.TestCase):
    def test_budget_is_met_and_threshold_is_minimal(self):
        rng = np.random.default_rng(42)
        y = np.r_[np.zeros(20_000), np.ones(500)].astype(int)
        raw = np.r_[rng.normal(0, 1, 20_000), rng.normal(3, 1, 500)]
        for dtype in (np.float32, np.float64):
            scores = raw.astype(dtype)
            benign = scores[y == 0]
            for budget in TARGET_FPR_BUDGETS:
                with self.subTest(dtype=dtype, budget=budget):
                    thr = find_threshold_for_fpr(y, scores, budget)
                    self.assertLessEqual(np.mean(benign >= thr), budget)
                    lower = np.nextafter(dtype(thr), dtype(-np.inf))
                    self.assertGreater(np.mean(benign >= lower), budget)

    def test_exact_budget_is_fully_used(self):
        y, scores = np.zeros(1000, int), np.arange(1000.0)
        thr = find_threshold_for_fpr(y, scores, 0.001)
        self.assertEqual(int((scores >= thr).sum()), 1)

    def test_float_rounding_of_budget(self):
        y, scores = np.zeros(100, int), np.arange(100.0)
        for k in range(1, 100):
            thr = find_threshold_for_fpr(y, scores, k / 100)
            self.assertEqual(int((scores >= thr).sum()), k)

    def test_ties_never_exceed_budget(self):
        y = np.zeros(1000, int)
        scores = np.zeros(1000, np.float32)
        scores[:50] = np.float32(0.7)
        thr = find_threshold_for_fpr(y, scores, 0.01)
        self.assertEqual(int((scores >= thr).sum()), 0)
        thr = find_threshold_for_fpr(y, scores, 0.05)
        self.assertEqual(int((scores >= thr).sum()), 50)

    def test_small_sample_allows_no_false_positives(self):
        y, scores = np.zeros(100, int), np.arange(100.0)
        thr = find_threshold_for_fpr(y, scores, 0.001)
        self.assertEqual(int((scores >= thr).sum()), 0)

    def test_attack_scores_do_not_affect_threshold(self):
        benign = np.arange(1000.0)
        only_benign = find_threshold_for_fpr(np.zeros(1000, int), benign)
        mixed = find_threshold_for_fpr(
            np.r_[np.zeros(1000, int), np.ones(3, int)], np.r_[benign, 1e6, -5, 10]
        )
        self.assertEqual(only_benign, mixed)

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            find_threshold_for_fpr([1, 1], [0.1, 0.2])
        for max_fpr in (0.0, 1.0, -0.1):
            with self.subTest(max_fpr=max_fpr), self.assertRaises(ValueError):
                find_threshold_for_fpr([0, 1], [0.1, 0.2], max_fpr)


class PerAttackRecallTest(unittest.TestCase):
    def test_breakdown(self):
        labels = ["BENIGN"] * 4 + ["PortScan"] * 2 + ["Botnet"] * 2
        preds = [0, 0, 0, 1, 1, 1, 1, 0]
        r = compute_per_attack_recall(labels, preds)
        self.assertEqual(list(r["per_attack"]), ["Botnet", "PortScan"])
        self.assertEqual(
            r["per_attack"]["Botnet"],
            {"n": 2, "detected": 1, "missed": 1, "recall": 0.5},
        )
        self.assertEqual(r["per_attack"]["PortScan"]["recall"], 1.0)
        self.assertEqual(r["macro_recall"], 0.75)
        self.assertEqual(
            r["benign"],
            {
                "n": 4,
                "false_positives": 1,
                "fpr": 0.25,
                "alerts_per_100k_benign": 25_000.0,
            },
        )
        json.dumps(r)

    def test_benign_label_ignores_case_and_spaces(self):
        r = compute_per_attack_recall([" Benign", "DoS"], [0, 1])
        self.assertEqual(list(r["per_attack"]), ["DoS"])
        self.assertEqual(r["benign"]["n"], 1)

    def test_only_benign_or_only_attacks(self):
        r = compute_per_attack_recall(["BENIGN"], [0])
        self.assertEqual(r["per_attack"], {})
        self.assertTrue(math.isnan(r["macro_recall"]))
        r = compute_per_attack_recall(["DoS"], [1])
        self.assertTrue(math.isnan(r["benign"]["fpr"]))

    def test_agrees_with_binary_metrics(self):
        rng = np.random.default_rng(0)
        labels = rng.choice(["BENIGN", "DoS", "PortScan"], 5000)
        scores = rng.random(5000)
        m = compute_binary_soc_metrics((labels != "BENIGN").astype(int), scores, 0.5)
        r = compute_per_attack_recall(labels, scores >= 0.5)
        detected = sum(row["detected"] for row in r["per_attack"].values())
        self.assertEqual(detected, m["confusion_matrix"]["tp"])
        self.assertEqual(r["benign"]["fpr"], m["fpr"])

    def test_rejects_bad_input(self):
        for labels, preds in (
            (["DoS"], [1, 0]),
            (["DoS", None], [1, 0]),
            (["DoS"], [0.7]),
        ):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                compute_per_attack_recall(labels, preds)


if __name__ == "__main__":
    unittest.main()
