"""ETL por dataset: descarga, limpieza, poda anti-leakage y splits.

Estado: stub. Contrato en `docs/03_logical_architecture.md`.

Uso previsto:
    python -m src.etl --dataset cicids2017
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from src.config import DEFAULT_DATASET, SUPPORTED_DATASETS

if TYPE_CHECKING:
    import pandas as pd


def download_dataset(dataset: str = DEFAULT_DATASET) -> Path:
    """Descarga los archivos crudos a `get_raw_dir(dataset)` y devuelve la ruta."""
    raise NotImplementedError


def load_raw(dataset: str = DEFAULT_DATASET) -> pd.DataFrame:
    """Lee los crudos y los adapta al esquema común (labels y familias canónicas)."""
    raise NotImplementedError


def clean_flows(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza columnas y elimina Inf, NaN y duplicados exactos."""
    raise NotImplementedError


def split_flows(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Separa en train/val/test por bloques temporales, sin partir sesiones.

    Requiere la metadata (IPs, puertos, timestamp): se ejecuta antes de la poda.
    """
    raise NotImplementedError


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina las columnas de `config.LEAKAGE_COLUMNS`."""
    raise NotImplementedError


def run_etl(dataset: str = DEFAULT_DATASET) -> dict[str, Path]:
    """Ejecuta el ETL completo y devuelve la ruta del parquet de cada split."""
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="ETL de flujos de red")
    parser.add_argument(
        "--dataset", choices=SUPPORTED_DATASETS, default=DEFAULT_DATASET
    )
    args = parser.parse_args()
    run_etl(args.dataset)


if __name__ == "__main__":
    main()
