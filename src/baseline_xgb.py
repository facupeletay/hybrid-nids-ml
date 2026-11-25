"""Capa supervisada: XGBoost / LightGBM, binario y multiclase.

Estado: stub. Contrato en `docs/03_logical_architecture.md`.

Uso previsto:
    python -m src.baseline_xgb --dataset cicids2017 --task binary
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any

from src.config import DEFAULT_DATASET, SUPPORTED_DATASETS

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

TASKS = ("binary", "multiclass")
BACKENDS = ("xgboost", "lightgbm")


def train_supervised(
    train: pd.DataFrame,
    val: pd.DataFrame,
    task: str = "binary",
    backend: str = "xgboost",
) -> Any:
    """Entrena con early stopping en Val y devuelve el modelo."""
    raise NotImplementedError


def score_supervised(model: Any, flows: pd.DataFrame) -> np.ndarray:
    """Score de sospecha por flujo, en [0, 1] (`1 - P(BENIGN)` en multiclase)."""
    raise NotImplementedError


def run(
    dataset: str = DEFAULT_DATASET, task: str = "binary", backend: str = "xgboost"
) -> dict[str, Any]:
    """Entrena, calibra el umbral en Val, evalúa en Test y persiste artefactos."""
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Clasificador supervisado")
    parser.add_argument(
        "--dataset", choices=SUPPORTED_DATASETS, default=DEFAULT_DATASET
    )
    parser.add_argument("--task", choices=TASKS, default="binary")
    parser.add_argument("--backend", choices=BACKENDS, default="xgboost")
    args = parser.parse_args()
    run(args.dataset, args.task, args.backend)


if __name__ == "__main__":
    main()
