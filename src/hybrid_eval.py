"""Evaluación híbrida y protocolo zero-day Leave-One-Attack-Out (LOAO).

Estado: stub. Contrato en `docs/03_logical_architecture.md`.

Uso previsto:
    python -m src.hybrid_eval --dataset cicids2017 --max-fpr 0.001
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any

from src.config import DEFAULT_DATASET, SUPPORTED_DATASETS, TARGET_FPR_BUDGETS

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


def calibrate_hybrid_thresholds(
    y_true_val: np.ndarray,
    supervised_val: np.ndarray,
    autoencoder_val: np.ndarray,
    max_fpr: float = TARGET_FPR_BUDGETS[0],
) -> tuple[float, float]:
    """Calibra ambos umbrales para que el FPR *conjunto* en Val sea <= `max_fpr`."""
    raise NotImplementedError


def combine_alerts(
    supervised_alerts: np.ndarray, autoencoder_alerts: np.ndarray
) -> np.ndarray:
    """Combina las alertas de ambas capas (OR lógico)."""
    raise NotImplementedError


def leave_one_attack_out(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    held_out_family: str,
    max_fpr: float = TARGET_FPR_BUDGETS[0],
) -> dict[str, Any]:
    """Excluye una familia de Train y Val, y mide su detección en Test."""
    raise NotImplementedError


def run(
    dataset: str = DEFAULT_DATASET, max_fpr: float = TARGET_FPR_BUDGETS[0]
) -> dict[str, Any]:
    """Evalúa el sistema híbrido y LOAO sobre todas las familias del dataset."""
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación híbrida y LOAO")
    parser.add_argument(
        "--dataset", choices=SUPPORTED_DATASETS, default=DEFAULT_DATASET
    )
    parser.add_argument("--max-fpr", type=float, default=TARGET_FPR_BUDGETS[0])
    args = parser.parse_args()
    run(args.dataset, args.max_fpr)


if __name__ == "__main__":
    main()
