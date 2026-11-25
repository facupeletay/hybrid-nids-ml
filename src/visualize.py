"""Figuras de evaluación, guardadas en `config.FIGURES_DIR`.

Estado: stub. Contrato en `docs/03_logical_architecture.md`.

Uso previsto:
    python -m src.visualize --dataset cicids2017
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.config import DEFAULT_DATASET, SUPPORTED_DATASETS

if TYPE_CHECKING:
    import numpy as np


def plot_pr_curves(
    y_true: np.ndarray, scores_by_model: dict[str, np.ndarray], output: Path
) -> Path:
    """Curvas Precision-Recall por modelo."""
    raise NotImplementedError


def plot_recall_at_fpr(
    y_true: np.ndarray, scores_by_model: dict[str, np.ndarray], output: Path
) -> Path:
    """Recall en función del FPR (eje log), con los presupuestos marcados."""
    raise NotImplementedError


def plot_reconstruction_error(
    errors: np.ndarray, families: np.ndarray, threshold: float, output: Path
) -> Path:
    """Distribución del error de reconstrucción, benigno vs. ataque."""
    raise NotImplementedError


def plot_detection_by_family(per_attack: dict[str, Any], output: Path) -> Path:
    """Recall por familia, a partir de `compute_per_attack_recall`."""
    raise NotImplementedError


def run(dataset: str = DEFAULT_DATASET) -> list[Path]:
    """Genera todas las figuras del dataset y devuelve sus rutas."""
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Figuras de evaluación")
    parser.add_argument(
        "--dataset", choices=SUPPORTED_DATASETS, default=DEFAULT_DATASET
    )
    args = parser.parse_args()
    run(args.dataset)


if __name__ == "__main__":
    main()
