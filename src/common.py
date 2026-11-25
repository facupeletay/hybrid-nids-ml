"""Utilidades compartidas por las capas de modelado.

No dependen de ningún framework: puntuar con una capa no exige instalar la otra.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.config import (
    DESTINATION_PORT_COLUMNS,
    TARGET_FPR_BUDGETS,
    get_feature_columns,
)
from src.etl import atomic_write
from src.metrics import find_threshold_for_fpr

logger = logging.getLogger(__name__)

NO_PORT_SUFFIX = "_no_dst_port"


# --------------------------------------------------------------------------
# Features y calibración
# --------------------------------------------------------------------------
def select_features(
    columns: Iterable[str], drop_dst_port: bool = False
) -> tuple[list[str], list[str]]:
    """Features del modelo y columnas quitadas por la ablación.

    Raises:
        ValueError: Si se pide quitar el puerto destino y no está: una ablación
            que no quita nada informaría un impacto nulo falso.
    """
    features = get_feature_columns(columns)
    if not drop_dst_port:
        return features, []
    dropped = [c for c in features if c in DESTINATION_PORT_COLUMNS]
    if not dropped:
        raise ValueError("No hay columna de puerto destino para quitar")
    return [c for c in features if c not in dropped], dropped


def calibrate_thresholds(
    y_val: np.ndarray,
    score_val: np.ndarray,
    budgets: Iterable[float] = TARGET_FPR_BUDGETS,
) -> dict[float, float]:
    """Umbral por presupuesto de FPR. Por construcción no recibe Test."""
    return {
        budget: find_threshold_for_fpr(y_val, score_val, budget) for budget in budgets
    }


# --------------------------------------------------------------------------
# Artefactos
# --------------------------------------------------------------------------
def normalize_tag(tag: str) -> str:
    """Valida un tag de artefacto y le antepone `_` si hace falta."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]*", tag):
        raise ValueError(f"tag inválido: {tag!r} (letras, dígitos, '_', '-' y '.')")
    return f"_{tag}" if tag and not tag.startswith("_") else tag


def in_dir(path: Path, directory: Path | None) -> Path:
    """Ubica el archivo en `directory`, si se indica otra carpeta."""
    return directory / path.name if directory else path


def to_strict_json(value: Any) -> Any:
    """Convierte a tipos JSON; lo no finito (métrica indefinida) pasa a `null`."""
    if isinstance(value, Mapping):
        return {str(key): to_strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_strict_json(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    """Escribe el reporte de forma atómica y como JSON estricto."""
    text = json.dumps(
        to_strict_json(report), indent=2, ensure_ascii=False, allow_nan=False
    )
    atomic_write(path, lambda tmp: tmp.write_text(text, encoding="utf-8"))
    logger.info("Reporte -> %s", path)


# --------------------------------------------------------------------------
# Tipos de argparse
# --------------------------------------------------------------------------
def positive_int(text: str) -> int:
    if (value := int(text)) < 1:
        raise argparse.ArgumentTypeError("debe ser un entero >= 1")
    return value


def positive_float(text: str) -> float:
    value = float(text)
    if not (math.isfinite(value) and value > 0):
        raise argparse.ArgumentTypeError("debe ser un número > 0")
    return value
