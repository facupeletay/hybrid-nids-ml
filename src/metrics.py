"""Métricas orientadas a SOC para detección de intrusiones.

Convenciones:
    - Etiqueta binaria: 0 = benigno, 1 = ataque.
    - Score: mayor = más sospechoso (probabilidad o error de reconstrucción).
    - Alerta: `score >= threshold`.
    - Una métrica indefinida (p. ej. Recall sin ataques) vale `nan`, no 0.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import BENIGN_LABEL

ALERT_RATE_BASE = 100_000


# --------------------------------------------------------------------------
# Validación
# --------------------------------------------------------------------------
def _as_binary(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} debe ser un vector 1-D no vacío")
    if not np.isin(array, (0, 1)).all():
        raise ValueError(f"{name} solo admite 0 (benigno) y 1 (ataque)")
    return array.astype(np.int8)


def _as_scores(values: ArrayLike, expected_size: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size != expected_size:
        raise ValueError(
            f"y_score debe ser 1-D de largo {expected_size}; llegó {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float64)
    if not np.isfinite(array).all():
        raise ValueError("y_score contiene NaN o Inf")
    return array


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else math.nan


# --------------------------------------------------------------------------
# Métricas
# --------------------------------------------------------------------------
def compute_binary_soc_metrics(
    y_true: ArrayLike, y_score: ArrayLike, threshold: float
) -> dict[str, Any]:
    """Métricas binarias en un punto de operación y métricas de ranking.

    Args:
        y_true: Etiquetas binarias.
        y_score: Scores de sospecha.
        threshold: Umbral calibrado en validación.

    Returns:
        Diccionario con `precision`, `recall` (TPR), `fpr`, `f1`, `roc_auc`,
        `pr_auc`, `confusion_matrix` (tn, fp, fn, tp), `alerts_per_100k_benign`
        y tamaños. `roc_auc` y `pr_auc` no dependen del umbral y valen `nan`
        si falta una de las clases.
    """
    y = _as_binary(y_true, "y_true")
    scores = _as_scores(y_score, y.size)
    if math.isnan(threshold):
        raise ValueError("threshold no puede ser NaN")

    alerts = scores >= threshold
    is_attack = y == 1
    tp = int(np.count_nonzero(alerts & is_attack))
    fp = int(np.count_nonzero(alerts & ~is_attack))
    fn = int(np.count_nonzero(~alerts & is_attack))
    tn = int(np.count_nonzero(~alerts & ~is_attack))

    n_attack, n_benign = tp + fn, tn + fp
    fpr = _ratio(fp, n_benign)
    has_both = n_attack > 0 and n_benign > 0

    return {
        "threshold": float(threshold),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, n_attack),
        "fpr": fpr,
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "roc_auc": float(roc_auc_score(y, scores)) if has_both else math.nan,
        "pr_auc": float(average_precision_score(y, scores)) if has_both else math.nan,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "alerts_per_100k_benign": fpr * ALERT_RATE_BASE,
        "n_benign": n_benign,
        "n_attack": n_attack,
    }


def find_threshold_for_fpr(
    y_true_val: ArrayLike, y_score_val: ArrayLike, max_fpr: float = 0.001
) -> float:
    """Umbral mínimo que garantiza `FPR <= max_fpr` en validación.

    Se calibra estrictamente en Val, nunca en Test. La garantía es empírica sobre
    esa muestra: en Test el FPR puede diferir por ruido muestral o drift.

    Solo intervienen los scores benignos, así que admite una validación sin
    ataques (caso Autoencoder). Con `n` benignos se toleran `k` falsos positivos,
    el mayor entero con `k / n <= max_fpr`, y el umbral es el primer valor
    representable por encima del (k+1)-ésimo score benigno más alto. Los empates
    quedan así del lado seguro. Si `n < 1 / max_fpr`, `k = 0`.

    El umbral se calcula en el dtype de los scores: con `float32` (XGBoost,
    PyTorch), un umbral de mayor precisión se redondearía al comparar y
    volvería a dejar pasar los empates. Val y Test deben compartir dtype.

    Args:
        y_true_val: Etiquetas binarias de validación.
        y_score_val: Scores de validación.
        max_fpr: Presupuesto de FPR, en (0, 1).

    Returns:
        Umbral para usar con `score >= threshold`.
    """
    if not 0.0 < max_fpr < 1.0:
        raise ValueError(f"max_fpr debe estar en (0, 1); llegó {max_fpr}")
    y = _as_binary(y_true_val, "y_true_val")
    scores = _as_scores(y_score_val, y.size)

    benign = scores[y == 0]
    n = benign.size
    if n == 0:
        raise ValueError("Se requieren flujos benignos para calibrar el FPR")

    # max_fpr * n puede quedar a un ulp del entero (0.29 * 100 = 28.999...).
    k = math.floor(max_fpr * n)
    if (k + 1) / n <= max_fpr:
        k += 1
    elif k / n > max_fpr:
        k -= 1

    limit = np.partition(benign, n - 1 - k)[n - 1 - k]
    return float(np.nextafter(limit, np.inf, dtype=scores.dtype))


def compute_per_attack_recall(
    attack_labels: ArrayLike,
    y_pred_binary: ArrayLike,
    benign_label: str = BENIGN_LABEL,
) -> dict[str, Any]:
    """Tasa de detección por familia de ataque y falsos positivos en benignos.

    Args:
        attack_labels: Familia de cada flujo; los benignos llevan `benign_label`.
        y_pred_binary: Predicción binaria (1 = alerta).
        benign_label: Etiqueta benigna (no distingue mayúsculas ni espacios).

    Returns:
        Diccionario con:
        - `per_attack`: `{familia: {n, detected, missed, recall}}`, ordenado
          por recall ascendente (los puntos ciegos primero).
        - `macro_recall`: promedio simple entre familias.
        - `benign`: `{n, false_positives, fpr, alerts_per_100k_benign}`.
    """
    y_pred = _as_binary(y_pred_binary, "y_pred_binary")
    labels = pd.Series(np.asarray(attack_labels, dtype=object))
    if len(labels) != y_pred.size:
        raise ValueError(
            f"attack_labels ({len(labels)}) e y_pred_binary ({y_pred.size}) "
            "difieren en largo"
        )
    if labels.isna().any():
        raise ValueError("attack_labels contiene valores nulos")

    labels = labels.astype(str).str.strip()
    is_benign = (labels.str.upper() == benign_label.strip().upper()).to_numpy()

    per_attack = {}
    attack_preds = pd.Series(y_pred[~is_benign])
    for family, preds in attack_preds.groupby(labels.to_numpy()[~is_benign]):
        n, detected = int(preds.size), int(preds.sum())
        per_attack[str(family)] = {
            "n": n,
            "detected": detected,
            "missed": n - detected,
            "recall": detected / n,
        }
    per_attack = dict(sorted(per_attack.items(), key=lambda kv: kv[1]["recall"]))

    n_benign = int(is_benign.sum())
    false_positives = int(y_pred[is_benign].sum())
    fpr = _ratio(false_positives, n_benign)
    recalls = [row["recall"] for row in per_attack.values()]

    return {
        "per_attack": per_attack,
        "macro_recall": float(np.mean(recalls)) if recalls else math.nan,
        "benign": {
            "n": n_benign,
            "false_positives": false_positives,
            "fpr": fpr,
            "alerts_per_100k_benign": fpr * ALERT_RATE_BASE,
        },
    }
