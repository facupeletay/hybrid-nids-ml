"""Capa no supervisada: Autoencoder en PyTorch entrenado solo con benignos.

Estado: stub. Contrato en `docs/03_logical_architecture.md`.

Uso previsto:
    python -m src.autoencoder --dataset cicids2017
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any

from src.config import DEFAULT_DATASET, SUPPORTED_DATASETS

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


def fit_preprocessor(benign_train: pd.DataFrame) -> Any:
    """Ajusta el escalado exclusivamente con el tráfico benigno de Train."""
    raise NotImplementedError


def train_autoencoder(
    benign_train: np.ndarray, benign_val: np.ndarray
) -> tuple[Any, dict[str, list[float]]]:
    """Entrena con early stopping sobre Val benigno; devuelve modelo e historia."""
    raise NotImplementedError


def reconstruction_error(model: Any, flows: np.ndarray) -> np.ndarray:
    """Error de reconstrucción por flujo (mayor = más anómalo)."""
    raise NotImplementedError


def run(dataset: str = DEFAULT_DATASET) -> dict[str, Any]:
    """Entrena, calibra el umbral en Val, evalúa en Test y persiste artefactos."""
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Autoencoder de tráfico benigno")
    parser.add_argument(
        "--dataset", choices=SUPPORTED_DATASETS, default=DEFAULT_DATASET
    )
    args = parser.parse_args()
    run(args.dataset)


if __name__ == "__main__":
    main()
