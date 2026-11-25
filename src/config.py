"""Configuración central: rutas dinámicas (Colab Drive vs. local) y constantes.

Importar este módulo no tiene efectos secundarios sobre el disco. Las carpetas
se crean solo al llamar a `ensure_directories()`.

Resolución de la raíz de almacenamiento (`STORAGE_ROOT`):
    1. `/content/drive/MyDrive/hybrid_nids`, si Drive está montado.
    2. La raíz del repositorio, en cualquier otro caso.
"""

from __future__ import annotations

import os
import re
import sys
import warnings
from collections.abc import Iterable
from pathlib import Path

# --------------------------------------------------------------------------
# Entorno
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DRIVE_MY_DRIVE = Path("/content/drive/MyDrive")
DRIVE_FOLDER_NAME = "hybrid_nids"


def is_drive_mounted(drive_root: Path = DRIVE_MY_DRIVE) -> bool:
    """Indica si Google Drive está montado (criterio de entorno Colab)."""
    return drive_root.is_dir()


def resolve_storage_root(drive_root: Path = DRIVE_MY_DRIVE) -> Path:
    """Devuelve la raíz donde persisten datos, modelos y métricas."""
    if is_drive_mounted(drive_root):
        return drive_root / DRIVE_FOLDER_NAME
    return PROJECT_ROOT


# --------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------
IN_COLAB = is_drive_mounted()
STORAGE_ROOT = resolve_storage_root()

DATA_DIR = STORAGE_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = STORAGE_ROOT / "models"
METRICS_DIR = STORAGE_ROOT / "metrics"
FIGURES_DIR = STORAGE_ROOT / "reports" / "figures"

# --------------------------------------------------------------------------
# Datasets y reproducibilidad
# --------------------------------------------------------------------------
SUPPORTED_DATASETS = ("cicids2017", "unsw_nb15")
DEFAULT_DATASET = "cicids2017"
SPLITS = ("train", "val", "test")
RANDOM_SEED = 42

# --------------------------------------------------------------------------
# Evaluación SOC
# --------------------------------------------------------------------------
# Presupuestos de FPR: 0.1 %, 0.5 % y 1 %. El primero es el objetivo operativo.
TARGET_FPR_BUDGETS = (0.001, 0.005, 0.01)

# --------------------------------------------------------------------------
# Esquema común (salida de cada adaptador de dataset)
# --------------------------------------------------------------------------
BENIGN_LABEL = "BENIGN"
TARGET_COLUMN = "is_attack"  # int8: 0 = benigno, 1 = ataque
ATTACK_LABEL_COLUMN = "attack_label"  # etiqueta original normalizada
ATTACK_FAMILY_COLUMN = "attack_family"  # familia canónica (unidad de LOAO)

# --------------------------------------------------------------------------
# Anti-leakage
# --------------------------------------------------------------------------
# Identifican hosts, sesiones o instantes del laboratorio. Pueden usarse como
# metadata para construir los splits, nunca como features.
LEAKAGE_COLUMNS = (
    # CIC-IDS2017 (CICFlowMeter v3 y nombres cortos de v4)
    "Flow ID",
    "Source IP",
    "Src IP",
    "Destination IP",
    "Dst IP",
    "Source Port",
    "Src Port",
    "Timestamp",
    # UNSW-NB15
    "id",  # índice de fila de las particiones oficiales
    "srcip",
    "dstip",
    "sport",
    "Stime",
    "Ltime",
    # Índice arrastrado por exportaciones a CSV
    "Unnamed: 0",
)

# No se podan por defecto, pero son propensas a *shortcut learning*: la topología
# del laboratorio las vuelve casi un proxy de la etiqueta. Requieren ablación.
SHORTCUT_PRONE_COLUMNS = (
    "Destination Port",
    "Dst Port",
    "dsport",
    "sttl",
    "dttl",
    "ct_state_ttl",
)


def normalize_column_name(name: str) -> str:
    """Reduce un nombre a minúsculas alfanuméricas (` Source IP` -> `sourceip`)."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


_LEAKAGE_KEYS = frozenset(normalize_column_name(c) for c in LEAKAGE_COLUMNS)


def find_leakage_columns(columns: Iterable[str]) -> list[str]:
    """Devuelve las columnas prohibidas presentes, con su nombre original.

    La comparación ignora mayúsculas, espacios y separadores: los CSV de
    CIC-IDS2017 traen encabezados con espacios iniciales.
    """
    return [c for c in columns if normalize_column_name(c) in _LEAKAGE_KEYS]


# --------------------------------------------------------------------------
# Helpers de rutas
# --------------------------------------------------------------------------
def validate_dataset(dataset: str) -> str:
    """Devuelve `dataset` si está soportado; si no, lanza `ValueError`."""
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Dataset no soportado: {dataset!r}. Opciones: {SUPPORTED_DATASETS}"
        )
    return dataset


def get_raw_dir(dataset: str = DEFAULT_DATASET) -> Path:
    """Carpeta de archivos crudos del dataset."""
    return RAW_DIR / validate_dataset(dataset)


def get_processed_dir(dataset: str = DEFAULT_DATASET) -> Path:
    """Carpeta de parquets procesados del dataset."""
    return PROCESSED_DIR / validate_dataset(dataset)


def get_split_path(dataset: str, split: str) -> Path:
    """Ruta del parquet de un split (`train`, `val` o `test`)."""
    if split not in SPLITS:
        raise ValueError(f"Split inválido: {split!r}. Opciones: {SPLITS}")
    return get_processed_dir(dataset) / f"{split}.parquet"


def ensure_directories() -> None:
    """Crea la estructura de carpetas de almacenamiento (idempotente)."""
    if not IN_COLAB and (
        "COLAB_RELEASE_TAG" in os.environ or "google.colab" in sys.modules
    ):
        warnings.warn(
            "Colab sin Drive montado: los artefactos se perderán al cerrar la "
            "sesión. Montá Drive antes de importar src.config.",
            stacklevel=2,
        )
    for dataset in SUPPORTED_DATASETS:
        get_raw_dir(dataset).mkdir(parents=True, exist_ok=True)
        get_processed_dir(dataset).mkdir(parents=True, exist_ok=True)
    for path in (MODELS_DIR, METRICS_DIR, FIGURES_DIR):
        path.mkdir(parents=True, exist_ok=True)


def describe() -> str:
    """Resumen legible de la configuración activa."""
    rows = {
        "entorno": "Colab + Drive" if IN_COLAB else "local",
        "STORAGE_ROOT": STORAGE_ROOT,
        "RAW_DIR": RAW_DIR,
        "PROCESSED_DIR": PROCESSED_DIR,
        "MODELS_DIR": MODELS_DIR,
        "METRICS_DIR": METRICS_DIR,
        "FIGURES_DIR": FIGURES_DIR,
        "SUPPORTED_DATASETS": SUPPORTED_DATASETS,
        "DEFAULT_DATASET": DEFAULT_DATASET,
        "RANDOM_SEED": RANDOM_SEED,
        "TARGET_FPR_BUDGETS": TARGET_FPR_BUDGETS,
    }
    return "\n".join(f"{key:<20} {value}" for key, value in rows.items())


if __name__ == "__main__":
    print(describe())
