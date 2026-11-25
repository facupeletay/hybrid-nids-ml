"""ETL por dataset: ingesta, limpieza, split temporal y poda anti-leakage.

Teoría y decisiones: `docs/theory/01_etl_and_anti_leakage.md`.

Uso:
    python -m src.etl --dataset cicids2017
    python -m src.etl --dataset cicids2017 --raw-dir /ruta/a/los/csv
    python -m src.etl --dataset cicids2017 --kaggle-dataset usuario/dataset

Los árboles consumen los parquets sin escalar. El `StandardScaler` se ajusta
exclusivamente con el tráfico benigno de Train y queda como referencia: el
Autoencoder comprime las colas antes de escalar y ajusta el suyo.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import logging
import os
import re
import shutil
import urllib.error
import urllib.request
import zipfile
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler

from src.config import (
    ATTACK_FAMILY_COLUMN,
    ATTACK_LABEL_COLUMN,
    BENIGN_LABEL,
    CAMPAIGN_COLUMN,
    DEFAULT_DATASET,
    METADATA_COLUMNS,
    SHORTCUT_PRONE_COLUMNS,
    SPLITS,
    SUPPORTED_DATASETS,
    TARGET_COLUMN,
    find_leakage_columns,
    get_etl_metadata_path,
    get_feature_columns,
    get_processed_dir,
    get_raw_dir,
    get_scaler_path,
    get_split_path,
    validate_dataset,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0.0"

# Metadata interna del split, común a todos los adaptadores. Se elimina siempre.
START_COLUMN = "_start"  # inicio del flujo, en segundos epoch
END_COLUMN = "_end"  # fin del flujo, en segundos epoch
FLOW_KEY_COLUMN = "_flow_key"  # 5-tupla bidireccional + día
INTERNAL_COLUMNS = (START_COLUMN, END_COLUMN, FLOW_KEY_COLUMN)

TRAIN, VAL, TEST = range(len(SPLITS))


class RawDataError(RuntimeError):
    """Los datos crudos faltan, no se pudieron obtener o no son utilizables."""


class ArtifactError(RuntimeError):
    """Los artefactos procesados faltan o son inconsistentes."""


@dataclass(frozen=True)
class EtlConfig:
    """Parámetros que determinan el resultado del ETL.

    Attributes:
        train_fraction: Fracción cronológica inicial de cada campaña.
        val_fraction: Fracción siguiente. El resto es Test.
        min_campaign_flows: Campañas más chicas van completas a Test.
        session_gap_seconds: Separación mínima entre flujos de la misma 5-tupla
            que caen en splits distintos. 65 s = 60 s de resolución del
            timestamp + 5 s, donde cae el 91.5 % de las continuaciones TCP
            medidas en el lunes (único día con resolución de 1 s).
    """

    train_fraction: float = 0.70
    val_fraction: float = 0.15
    min_campaign_flows: int = 50
    session_gap_seconds: float = 65.0

    def __post_init__(self) -> None:
        if not (
            self.train_fraction > 0
            and self.val_fraction > 0
            and self.train_fraction + self.val_fraction < 1
        ):
            raise ValueError("Las fracciones de Train y Val deben sumar menos de 1")


@dataclass
class EtlArtifacts:
    """Resultado de `load_artifacts`."""

    splits: dict[str, pd.DataFrame]
    scaler: StandardScaler
    metadata: dict[str, Any]
    feature_columns: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Archivos
# --------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, write: Callable[[Path], Any]) -> None:
    """Escribe en un temporal de la misma carpeta y renombra al final."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def download_file(url: str, target: Path, timeout: float = 60.0) -> Path:
    """Descarga `url` a un `.part` reanudable y renombra al completar.

    Raises:
        RawDataError: Si la red falla o la respuesta queda incompleta. El
            `.part` se conserva para reanudar en la próxima ejecución.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    offset = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}

    try:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            # Un 200 ante un Range significa que el servidor reenvía todo.
            resumed = response.status == 206
            length = response.headers.get("Content-Length")
            with part.open("ab" if resumed else "wb") as handle:
                shutil.copyfileobj(response, handle, 1 << 20)
            expected = (offset if resumed else 0) + int(length) if length else None
    except urllib.error.HTTPError as error:
        if error.code != 416:  # 416: el .part ya estaba completo
            raise RawDataError(f"HTTP {error.code} al descargar {url}") from error
        expected = None
    except (OSError, http.client.HTTPException) as error:
        raise RawDataError(f"No se pudo descargar {url}: {error}") from error

    if expected is not None and part.stat().st_size != expected:
        raise RawDataError(
            f"Descarga incompleta de {url}: {part.stat().st_size} de {expected} "
            "bytes. Reejecutá para reanudar."
        )
    os.replace(part, target)
    return target


# --------------------------------------------------------------------------
# Adaptadores
# --------------------------------------------------------------------------
class DatasetAdapter(ABC):
    """Todo lo específico de un dataset. El resto del ETL es genérico."""

    name: str

    @abstractmethod
    def obtain(self, raw_dir: Path, kaggle_dataset: str | None = None) -> None:
        """Deja los archivos crudos en `raw_dir`; no hace nada si ya están."""

    @abstractmethod
    def find_sources(self, raw_dir: Path) -> list[Path]:
        """Ubica y valida los archivos crudos, en orden determinista."""

    @abstractmethod
    def load(self, path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Lee y limpia un archivo.

        Returns:
            Flujos con features `float32`, `METADATA_COLUMNS`,
            `INTERNAL_COLUMNS` e identificadores originales, y un reporte de
            limpieza. Las reglas no pueden depender de estadísticas globales.
        """


class CicIds2017Adapter(DatasetAdapter):
    """CIC-IDS2017, distribución `GeneratedLabelledFlows` (TrafficLabelling)."""

    name = "cicids2017"
    url = (
        "http://cicresearch.ca/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/"
        "CSVs/GeneratedLabelledFlows.zip"
    )

    # Prefijo del archivo -> (campaign_day, fecha de captura).
    files = {
        "monday-workinghours": ("monday", date(2017, 7, 3)),
        "tuesday-workinghours": ("tuesday", date(2017, 7, 4)),
        "wednesday-workinghours": ("wednesday", date(2017, 7, 5)),
        "thursday-workinghours-morning-webattacks": (
            "thursday_morning_webattacks",
            date(2017, 7, 6),
        ),
        "thursday-workinghours-afternoon-infilteration": (
            "thursday_afternoon_infiltration",
            date(2017, 7, 6),
        ),
        "friday-workinghours-morning": ("friday_morning", date(2017, 7, 7)),
        "friday-workinghours-afternoon-portscan": (
            "friday_afternoon_portscan",
            date(2017, 7, 7),
        ),
        "friday-workinghours-afternoon-ddos": (
            "friday_afternoon_ddos",
            date(2017, 7, 7),
        ),
    }

    required_columns = (
        "Source IP",
        "Source Port",
        "Destination IP",
        "Destination Port",
        "Protocol",
        "Timestamp",
        "Flow Duration",
        "Label",
    )
    text_columns = ("Flow ID", "Source IP", "Destination IP", "Timestamp", "Label")

    # -1 = "sin ventana TCP" (UDP o dirección sin paquetes): valor legítimo.
    sentinel_columns = ("Init_Win_bytes_forward", "Init_Win_bytes_backward")

    families = {
        "BENIGN": BENIGN_LABEL,
        "DoS Hulk": "DoS",
        "DoS GoldenEye": "DoS",
        "DoS slowloris": "DoS",
        "DoS Slowhttptest": "DoS",
        "Heartbleed": "DoS",
        "DDoS": "DDoS",
        "PortScan": "PortScan",
        "FTP-Patator": "BruteForce",
        "SSH-Patator": "BruteForce",
        "Web Attack - Brute Force": "WebAttack",
        "Web Attack - XSS": "WebAttack",
        "Web Attack - Sql Injection": "WebAttack",
        "Bot": "Bot",
        "Infiltration": "Infiltration",
    }

    # La captura va de ~08:40 a ~17:10 en reloj de 12 h sin AM/PM.
    first_hour = 7
    last_hour = 19

    _timestamp_re = re.compile(
        r"^(\d{1,4})[/-](\d{1,2})[/-](\d{1,4})[ T]+(\d{1,2}):(\d{2})"
        r"(?::(\d{2})(?:\.\d+)?)?\s*([AaPp][Mm])?$"
    )

    # ---------------------------------------------------------------- ingesta
    def obtain(self, raw_dir: Path, kaggle_dataset: str | None = None) -> None:
        if not self._missing(self._candidates(raw_dir)):
            logger.info("Los %d CSV ya están en %s", len(self.files), raw_dir)
            return
        if kaggle_dataset:
            self._obtain_from_kaggle(raw_dir, kaggle_dataset)
        else:
            self._obtain_from_unb(raw_dir)

    def _obtain_from_unb(self, raw_dir: Path) -> None:
        archive = raw_dir / "GeneratedLabelledFlows.zip"
        downloaded = not archive.exists()
        if downloaded:
            logger.info("Descargando %s", self.url)
            download_file(self.url, archive)
        if not zipfile.is_zipfile(archive):
            with archive.open("rb") as handle:
                is_html = b"<html" in handle.read(4096).lower()
            if downloaded:
                archive.unlink()
            reason = (
                "devolvió una página HTML: UNB exige completar un formulario "
                "de registro"
                if is_html
                else "no devolvió un ZIP válido"
            )
            origin = self.url if downloaded else f"{archive}: la fuente"
            raise RawDataError(
                f"{origin} {reason}. Alternativas: (1) registrarse en "
                "https://www.unb.ca/cic/datasets/ids-2017.html, bajar "
                f"GeneratedLabelledFlows.zip a {raw_dir} y reejecutar; "
                "(2) --raw-dir con los CSV ya descargados; (3) --kaggle-dataset "
                "con un mirror de TrafficLabelling."
            )
        self._extract(archive, raw_dir)
        # Un ZIP que dejó el usuario no se borra.
        if downloaded:
            archive.unlink()

    def _extract(self, archive: Path, raw_dir: Path) -> None:
        try:
            with zipfile.ZipFile(archive) as bundle:
                for info in bundle.infolist():
                    # Solo el nombre base: ignora rutas internas del ZIP.
                    name = Path(info.filename).name
                    if info.is_dir() or self._file_key(name) is None:
                        continue

                    def write(tmp: Path, info: zipfile.ZipInfo = info) -> None:
                        with bundle.open(info) as src, tmp.open("wb") as dst:
                            shutil.copyfileobj(src, dst, 1 << 20)

                    atomic_write(raw_dir / name, write)
        except zipfile.BadZipFile as error:
            raise RawDataError(f"{archive} está corrupto: {error}. Borralo y reejecutá") from error

    def _obtain_from_kaggle(self, raw_dir: Path, slug: str) -> None:
        import kagglehub

        logger.info("Descargando %s vía kagglehub", slug)
        for source in self.find_sources(Path(kagglehub.dataset_download(slug))):
            atomic_write(
                raw_dir / source.name,
                lambda tmp, source=source: shutil.copyfile(source, tmp),
            )

    # ----------------------------------------------------------- localización
    def _file_key(self, name: str) -> str | None:
        stem = name.lower()
        if not stem.endswith(".csv"):
            return None
        return next((key for key in self.files if stem.startswith(key)), None)

    def _candidates(self, raw_dir: Path) -> dict[str, list[Path]]:
        found: dict[str, list[Path]] = {key: [] for key in self.files}
        if raw_dir.is_dir():
            for path in sorted(raw_dir.rglob("*")):
                key = self._file_key(path.name)
                if key and path.is_file():
                    found[key].append(path)
        return found

    @staticmethod
    def _missing(candidates: dict[str, list[Path]]) -> list[str]:
        return [key for key, paths in candidates.items() if not paths]

    def find_sources(self, raw_dir: Path) -> list[Path]:
        candidates = self._candidates(raw_dir)
        if missing := self._missing(candidates):
            raise RawDataError(f"Faltan CSV de CIC-IDS2017 en {raw_dir}: {missing}")

        sources = []
        for key, paths in candidates.items():
            # Algunos mirrors traen juntas ambas distribuciones.
            usable = [p for p in paths if not self._missing_columns(p)]
            if not usable:
                self._raise_missing_columns(paths[0])
            if len(usable) > 1:
                raise RawDataError(f"Hay más de un CSV para {key!r}: {usable}")
            sources.append(usable[0])
        return sources

    def _missing_columns(self, path: Path) -> list[str]:
        header = pd.read_csv(path, nrows=0, encoding="latin-1")
        present = {c.strip() for c in header.columns}
        present |= {c.removeprefix("ï»¿") for c in present}  # BOM leído en latin-1
        return [c for c in self.required_columns if c not in present]

    def _raise_missing_columns(self, path: Path) -> None:
        raise RawDataError(
            f"{path} no trae {self._missing_columns(path)}. Parece ser "
            "MachineLearningCSV, que elimina IPs, puerto de origen y Timestamp: "
            "sin ellos no se puede ordenar cada campaña en el tiempo ni evitar "
            "que una sesión quede repartida entre Train y Test, y solo queda un "
            "split aleatorio con fuga. Usá GeneratedLabelledFlows "
            "(TrafficLabelling)."
        )

    # ---------------------------------------------------------------- lectura
    @staticmethod
    def _read_csv(path: Path) -> pd.DataFrame:
        # El CSV de WebAttacks usa el byte \x96. latin-1 nunca falla: va último.
        for encoding in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                return pd.read_csv(path, encoding=encoding, low_memory=False)
            except UnicodeDecodeError:
                continue
        raise AssertionError("latin-1 decodifica cualquier byte")

    @staticmethod
    def normalize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Quita espacios y columnas repetidas (`X` y `X.1`), por nombre.

        Decidir por nombre, y no por igualdad de valores, evita que Val o Test
        influyan en qué columnas sobreviven.
        """
        names = [str(c).strip() for c in df.columns]
        seen: set[str] = set()
        keep, dropped = [], []
        for name in names:
            base = re.sub(r"\.\d+$", "", name)
            is_repeat = name in seen or (base != name and base in names)
            keep.append(not is_repeat)
            if is_repeat:
                dropped.append(name)
            seen.add(name)
        out = df.loc[:, keep].copy()
        out.columns = [n for n, k in zip(names, keep) if k]
        return out, dropped

    @classmethod
    def normalize_label(cls, label: str) -> str:
        """Etiqueta canónica; corrige el guion corrupto de `Web Attack`."""
        text = re.sub(r"\s+", " ", str(label)).strip()
        # Entre "Web Attack" y el nombre puede haber \x96, –, � o "ï¿½".
        text = re.sub(r"(?i)^web attack[^A-Za-z]+", "Web Attack - ", text)
        by_lower = {known.lower(): known for known in cls.families}
        try:
            return by_lower[text.lower()]
        except KeyError:
            raise RawDataError(f"Etiqueta desconocida: {label!r}") from None

    @classmethod
    def parse_timestamp(cls, text: str, expected: date) -> datetime:
        """Reconstruye el instante real a partir del timestamp ambiguo.

        El orden día/mes se resuelve contra la fecha conocida del archivo. Sin
        AM/PM, una hora menor a `first_hour` corresponde a la tarde.
        """
        match = cls._timestamp_re.match(str(text).strip())
        if not match:
            raise RawDataError(f"Timestamp ilegible: {text!r}")
        a, b, c, hour, minute, second = (int(g or 0) for g in match.groups()[:6])
        meridiem = (match.group(7) or "").lower()

        if len(match.group(1)) == 4:
            candidates = [(a, b, c)]
        else:
            year = c + 2000 if c < 100 else c
            candidates = [(year, b, a), (year, a, b)]
        if (expected.year, expected.month, expected.day) not in candidates:
            raise RawDataError(f"Timestamp {text!r} no es del {expected}")

        if meridiem:
            hour = hour % 12 + (12 if meridiem == "pm" else 0)
        elif hour < cls.first_hour:
            hour += 12
        if not cls.first_hour <= hour < cls.last_hour:
            raise RawDataError(f"Timestamp fuera de la captura: {text!r}")
        return datetime(
            expected.year, expected.month, expected.day, hour, minute, second
        )

    def load(self, path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
        key = self._file_key(path.name)
        if key is None:
            raise RawDataError(f"Archivo no reconocido: {path.name}")
        campaign, expected_date = self.files[key]

        df, repeated = self.normalize_columns(self._read_csv(path))
        report: dict[str, Any] = {
            "campaign_day": campaign,
            "rows_read": len(df),
            "repeated_columns_dropped": repeated,
        }

        df = df.dropna(how="all")
        report["empty_rows_dropped"] = report["rows_read"] - len(df)
        if df["Label"].isna().any():
            raise RawDataError(f"{path.name} tiene filas con datos y sin Label")

        labels = {raw: self.normalize_label(raw) for raw in df["Label"].unique()}
        df["Label"] = df["Label"].map(labels)

        # Metadata incluida: solo el mismo flujo exportado dos veces.
        duplicated = df.duplicated()
        report["exact_duplicates_dropped"] = _count_by(df.loc[duplicated, "Label"])
        df = df[~duplicated].reset_index(drop=True)

        df = self._clean_numeric(df, report)

        stamps = {
            text: self.parse_timestamp(text, expected_date)
            for text in df["Timestamp"].unique()
        }
        # Sin zona horaria: se usa UTC solo como referencia para restar.
        epochs = {
            text: stamp.replace(tzinfo=timezone.utc).timestamp()
            for text, stamp in stamps.items()
        }
        start = df["Timestamp"].map(epochs).astype(np.float64)
        duration = df["Flow Duration"].astype(np.float64) / 1e6
        family = df["Label"].map(self.families)

        extra = pd.DataFrame(
            {
                TARGET_COLUMN: (family != BENIGN_LABEL).astype(np.int8),
                ATTACK_FAMILY_COLUMN: family,
                ATTACK_LABEL_COLUMN: df["Label"],
                CAMPAIGN_COLUMN: campaign,
                START_COLUMN: start,
                END_COLUMN: start + duration,
                FLOW_KEY_COLUMN: self._flow_keys(df, expected_date),
            },
            index=df.index,
        )
        report["rows_kept"] = len(df)
        return pd.concat([df.drop(columns="Label"), extra], axis=1), report

    def _clean_numeric(
        self, df: pd.DataFrame, report: dict[str, Any]
    ) -> pd.DataFrame:
        """Aplica la política de valores inválidos (sin estadísticas globales).

        1. `±Inf` -> `NaN`; las filas con algún `NaN` se eliminan.
        2. Duración, contadores o tasas negativos: se elimina la fila.
        3. IAT negativo con duración válida: se lleva a 0 (microsegundos de
           reordenamiento de paquetes, no un flujo corrupto).
        4. `-1` en `sentinel_columns`: se conserva.
        """
        text = [c for c in self.text_columns if c in df.columns]
        values = df.drop(columns=text).apply(pd.to_numeric, errors="coerce")
        values = values.astype(np.float64)

        infinite = np.isinf(values)
        report["infinite_values"] = _nonzero(infinite.sum())
        values = values.mask(infinite)
        report["nan_values"] = _nonzero(values.isna().sum())
        has_nan = values.isna().any(axis=1)

        iat = [c for c in values if "IAT" in c]
        strict = [c for c in values if c not in iat and c not in self.sentinel_columns]
        negative = values[strict] < 0
        report["negative_values"] = _nonzero(negative.sum())
        has_negative = negative.any(axis=1) & ~has_nan

        valid = ~(has_nan | has_negative)
        report["nan_or_inf_rows_dropped"] = _count_by(df.loc[has_nan, "Label"])
        report["negative_rows_dropped"] = _count_by(df.loc[has_negative, "Label"])

        values = values[valid]
        clipped = values[iat] < 0
        report["negative_iat_clipped"] = _nonzero(clipped.sum())
        report["negative_iat_rows"] = _count_by(
            df.loc[valid, "Label"][clipped.any(axis=1)]
        )
        values[iat] = values[iat].clip(lower=0)

        cleaned = pd.concat(
            [df.loc[valid, text], values.astype(np.float32)], axis=1
        )
        return cleaned.reset_index(drop=True)

    @staticmethod
    def _flow_keys(df: pd.DataFrame, day: date) -> np.ndarray:
        """Clave de 64 bits de la 5-tupla, igual en ambos sentidos."""

        def endpoint(side: str) -> pd.Series:
            port = df[f"{side} Port"].astype(np.int64).astype(str)
            return df[f"{side} IP"].astype(str) + ":" + port

        a, b = endpoint("Source"), endpoint("Destination")
        forward = a <= b
        parts = pd.DataFrame(
            {
                "day": day.isoformat(),
                "low": a.where(forward, b),
                "high": b.where(forward, a),
                "protocol": df["Protocol"].astype(np.int64),
            }
        )
        return pd.util.hash_pandas_object(parts, index=False).to_numpy()


class UnswNb15Adapter(DatasetAdapter):
    """UNSW-NB15. Pendiente: solo debe implementar la interfaz."""

    name = "unsw_nb15"

    def obtain(self, raw_dir: Path, kaggle_dataset: str | None = None) -> None:
        raise NotImplementedError("El adaptador de unsw_nb15 no está implementado")

    def find_sources(self, raw_dir: Path) -> list[Path]:
        raise NotImplementedError("El adaptador de unsw_nb15 no está implementado")

    def load(self, path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
        raise NotImplementedError("El adaptador de unsw_nb15 no está implementado")


ADAPTERS: dict[str, DatasetAdapter] = {
    adapter.name: adapter for adapter in (CicIds2017Adapter(), UnswNb15Adapter())
}


def _count_by(labels: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in labels.value_counts().sort_index().items()}


def _nonzero(counts: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in counts.items() if v}


# --------------------------------------------------------------------------
# Ingesta
# --------------------------------------------------------------------------
def download_dataset(
    dataset: str = DEFAULT_DATASET, kaggle_dataset: str | None = None
) -> Path:
    """Deja los crudos en `get_raw_dir(dataset)` y devuelve la ruta."""
    raw_dir = get_raw_dir(dataset)
    raw_dir.mkdir(parents=True, exist_ok=True)
    ADAPTERS[dataset].obtain(raw_dir, kaggle_dataset)
    return raw_dir


def load_raw(
    dataset: str, raw_dir: Path
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Lee y limpia todos los archivos; devuelve flujos y reporte por archivo."""
    adapter = ADAPTERS[validate_dataset(dataset)]
    frames, reports = [], []
    for path in adapter.find_sources(raw_dir):
        frame, report = adapter.load(path)
        report = {
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            **report,
        }
        logger.info(
            "%s: %d leídas, %d vacías, %d conservadas",
            path.name,
            report["rows_read"],
            report["empty_rows_dropped"],
            report["rows_kept"],
        )
        frames.append(frame)
        reports.append(report)
    columns = set(frames[0].columns)
    for frame, report in zip(frames, reports):
        if set(frame.columns) != columns:
            diff = sorted(columns ^ set(frame.columns))
            raise RawDataError(f"{report['file']} difiere en columnas: {diff}")
    return pd.concat(frames, ignore_index=True), reports


# --------------------------------------------------------------------------
# Split temporal
# --------------------------------------------------------------------------
def find_cut_points(
    start: pd.Series, config: EtlConfig
) -> tuple[float, float] | None:
    """Instantes donde empiezan Val y Test, o `None` si no se puede dividir.

    Los cortes caen siempre entre dos timestamps distintos, lo más cerca posible
    de las fracciones pedidas: los empates nunca se reparten.
    """
    counts = start.value_counts().sort_index()
    if len(counts) < len(SPLITS):
        return None
    times = counts.index.to_numpy()
    cumulative = counts.cumsum().to_numpy() / counts.sum()
    train_end = config.train_fraction
    val_end = config.train_fraction + config.val_fraction

    i = int(np.argmin(np.abs(cumulative[:-2] - train_end)))
    j = i + 1 + int(np.argmin(np.abs(cumulative[i + 1 : -1] - val_end)))
    return float(times[i + 1]), float(times[j + 1])


def assign_temporal_split(
    df: pd.DataFrame, config: EtlConfig
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Asigna cada flujo a un split, en orden cronológico por campaña.

    Una campaña es la combinación `(campaign_day, label_raw)`. Las que tienen
    menos de `min_campaign_flows` flujos, o menos de tres timestamps distintos,
    van completas a Test.

    Returns:
        Vector con 0 (train), 1 (val) o 2 (test), y la regla aplicada a cada
        campaña.
    """
    split = np.full(len(df), TEST, dtype=np.int8)
    start = df[START_COLUMN].to_numpy()
    rules = []
    groups = df.groupby([CAMPAIGN_COLUMN, ATTACK_LABEL_COLUMN], sort=True).indices
    for (campaign, label), rows in groups.items():
        rule: dict[str, Any] = {
            "campaign_day": campaign,
            "label_raw": label,
            "flows": len(rows),
        }
        cuts = None
        if len(rows) < config.min_campaign_flows:
            rule["rule"] = "test_only_small_campaign"
        else:
            cuts = find_cut_points(pd.Series(start[rows]), config)
            rule["rule"] = "test_only_few_timestamps"
        if cuts is not None:
            val_start, test_start = cuts
            times = start[rows]
            split[rows] = np.select(
                [times < val_start, times < test_start], [TRAIN, VAL], TEST
            )
            rule["rule"] = "chronological"
            rule["val_start"] = _to_iso(val_start)
            rule["test_start"] = _to_iso(test_start)
        for code, name in enumerate(SPLITS):
            rule[f"{name}_flows"] = int(np.count_nonzero(split[rows] == code))
        rules.append(rule)
    return split, rules


def _to_iso(epoch: float) -> str:
    stamp = datetime.fromtimestamp(epoch, tz=timezone.utc)
    return stamp.replace(tzinfo=None).isoformat()


def find_boundary_flows(
    df: pd.DataFrame, split: np.ndarray, gap_seconds: float
) -> np.ndarray:
    """Marca los flujos a purgar para no partir sesiones en la frontera.

    Se purga un flujo si su misma 5-tupla aparece en un split *posterior*
    (train < val < test) a `gap_seconds` o menos. Se sacrifica siempre el lado
    anterior: Test nunca se toca.

    El criterio es por pares y no por cadenas. DoS Hulk reutiliza puertos y
    encadena toda su campaña en "sesiones" contiguas: exigir la cadena completa
    en un único split vaciaría Train.
    """
    order = np.lexsort((df[START_COLUMN].to_numpy(), df[FLOW_KEY_COLUMN].to_numpy()))
    work = pd.DataFrame(
        {
            "key": df[FLOW_KEY_COLUMN].to_numpy()[order],
            "start": df[START_COLUMN].to_numpy()[order],
            "end": df[END_COLUMN].to_numpy()[order],
            "split": split[order],
        }
    )
    purge = np.zeros(len(work), dtype=bool)
    # Primero contra Test; luego Train contra lo que quedó de Val y Test.
    for code in (TEST, VAL):
        alive = work[~purge]
        later = alive["split"] >= code
        next_start = alive["start"].where(later).groupby(alive["key"]).bfill()
        last_end = alive["end"].where(later).groupby(alive["key"]).cummax()
        last_end = last_end.groupby(alive["key"]).ffill()
        close = (next_start - alive["end"] <= gap_seconds) | (
            alive["start"] - last_end <= gap_seconds
        )
        hits = alive.index[~later & close]
        purge[hits] = True

    out = np.zeros(len(df), dtype=bool)
    out[order] = purge
    return out


def split_flows(
    df: pd.DataFrame, config: EtlConfig | None = None
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Separa en train/val/test; requiere `INTERNAL_COLUMNS` (antes de podar)."""
    config = config or EtlConfig()
    split, rules = assign_temporal_split(df, config)
    purge = find_boundary_flows(df, split, config.session_gap_seconds)

    info: dict[str, Any] = {"campaigns": rules, "session_purge": {}}
    labels = df[ATTACK_LABEL_COLUMN]
    frames = {}
    for code, name in enumerate(SPLITS):
        members = split == code
        info["session_purge"][name] = _count_by(labels[members & purge])
        frame = df[members & ~purge]
        frames[name] = frame.sort_values(
            [CAMPAIGN_COLUMN, START_COLUMN], kind="stable"
        ).reset_index(drop=True)

    info["emptied_by_purge"] = _campaigns_emptied_by_purge(frames, rules)
    for item in info["emptied_by_purge"]:
        logger.warning("La purga vació %s", item)
    return frames, info


def _campaigns_emptied_by_purge(
    frames: dict[str, pd.DataFrame], rules: Sequence[dict[str, Any]]
) -> list[dict[str, str]]:
    emptied = []
    for name, frame in frames.items():
        present = set(zip(frame[CAMPAIGN_COLUMN], frame[ATTACK_LABEL_COLUMN]))
        for rule in rules:
            campaign = (rule["campaign_day"], rule["label_raw"])
            if rule[f"{name}_flows"] and campaign not in present:
                emptied.append(
                    {"split": name, "campaign_day": campaign[0], "label_raw": campaign[1]}
                )
    return emptied


# --------------------------------------------------------------------------
# Poda y escalado
# --------------------------------------------------------------------------
def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina identificadores, timestamps y la metadata interna del split."""
    internal = [c for c in INTERNAL_COLUMNS if c in df.columns]
    return df.drop(columns=[*find_leakage_columns(df.columns), *internal])


def find_constant_columns(train: pd.DataFrame) -> list[str]:
    """Features con varianza 0 en Train."""
    features = train[get_feature_columns(train.columns)]
    return features.columns[features.max() == features.min()].tolist()


def fit_benign_scaler(train: pd.DataFrame) -> StandardScaler:
    """Ajusta el scaler exclusivamente con los flujos benignos de Train.

    `StandardScaler` y no `RobustScaler`: muchas features son 0 en más del 75 %
    del tráfico benigno, su IQR es 0 y quedarían sin escalar.
    """
    benign = train.loc[
        train[TARGET_COLUMN] == 0, get_feature_columns(train.columns)
    ]
    if benign.empty:
        raise ValueError("Train no tiene tráfico benigno para ajustar el scaler")
    return StandardScaler().fit(benign)


def count_train_twins(frames: dict[str, pd.DataFrame]) -> dict[str, dict[str, Any]]:
    """Flujos de Val/Test cuyo vector de features ya aparece en Train.

    No se eliminan: son tráfico real repetitivo. Se informan porque inflan el
    recall de las familias más mecánicas.
    """
    features = get_feature_columns(frames["train"].columns)

    def row_hashes(frame: pd.DataFrame) -> np.ndarray:
        return pd.util.hash_pandas_object(frame[features], index=False).to_numpy()

    known = np.unique(row_hashes(frames["train"]))
    twins: dict[str, dict[str, Any]] = {}
    for name in ("val", "test"):
        frame = frames[name]
        is_twin = pd.Series(np.isin(row_hashes(frame), known))
        grouped = is_twin.groupby(frame[ATTACK_LABEL_COLUMN].to_numpy())
        twins[name] = {
            str(label): {"flows": int(rows.size), "twins": int(rows.sum())}
            for label, rows in grouped
        }
    return twins


# --------------------------------------------------------------------------
# Orquestación
# --------------------------------------------------------------------------
def _artifact_paths(dataset: str, output_dir: Path | None) -> dict[str, Path]:
    paths = {split: get_split_path(dataset, split) for split in SPLITS}
    paths["scaler"] = get_scaler_path(dataset)
    paths["metadata"] = get_etl_metadata_path(dataset)
    base = output_dir or get_processed_dir(dataset)
    return {name: base / path.name for name, path in paths.items()}


def _config_hash(dataset: str, config: EtlConfig) -> str:
    payload = {"dataset": dataset, "schema": SCHEMA_VERSION, **asdict(config)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def run_etl(
    dataset: str = DEFAULT_DATASET,
    raw_dir: Path | None = None,
    kaggle_dataset: str | None = None,
    output_dir: Path | None = None,
    config: EtlConfig | None = None,
) -> dict[str, Path]:
    """Ejecuta el ETL completo.

    Args:
        dataset: Uno de `SUPPORTED_DATASETS`.
        raw_dir: CSV ya descargados. Si se omite, usa `get_raw_dir(dataset)` y
            descarga lo que falte.
        kaggle_dataset: Mirror de Kaggle a usar en lugar de la fuente oficial.
        output_dir: Destino; por defecto `get_processed_dir(dataset)`.
        config: Parámetros del ETL.

    Returns:
        Ruta de cada artefacto.
    """
    validate_dataset(dataset)
    config = config or EtlConfig()
    if raw_dir is None:
        raw_dir = download_dataset(dataset, kaggle_dataset)

    flows, reports = load_raw(dataset, Path(raw_dir))
    frames, split_info = split_flows(flows, config)
    del flows

    leakage = find_leakage_columns(frames["train"].columns)
    frames = {name: drop_leakage_columns(frame) for name, frame in frames.items()}
    constant = find_constant_columns(frames["train"])
    features = [
        c for c in get_feature_columns(frames["train"].columns) if c not in constant
    ]
    columns = [*features, *METADATA_COLUMNS]
    frames = {name: frame[columns] for name, frame in frames.items()}
    for frame in frames.values():
        _check_frame(frame)

    scaler = fit_benign_scaler(frames["train"])
    paths = _artifact_paths(dataset, output_dir)

    for name in SPLITS:
        frame = frames[name]
        atomic_write(
            paths[name],
            lambda tmp, frame=frame: frame.to_parquet(
                tmp, index=False, compression="zstd"
            ),
        )
        logger.info("%s: %d flujos -> %s", name, len(frame), paths[name])
    atomic_write(paths["scaler"], lambda tmp: joblib.dump(scaler, tmp))

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": asdict(config),
        "config_hash": _config_hash(dataset, config),
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "sources": reports,
        "split": split_info,
        "columns": {
            "features": features,
            "metadata": list(METADATA_COLUMNS),
            "dropped_leakage": leakage,
            "dropped_constant_in_train": constant,
            "shortcut_prone": [c for c in features if c in SHORTCUT_PRONE_COLUMNS],
        },
        "counts": {
            name: {
                "total": len(frame),
                "by_family": _count_by(frame[ATTACK_FAMILY_COLUMN]),
                "by_label": _count_by(frame[ATTACK_LABEL_COLUMN]),
            }
            for name, frame in frames.items()
        },
        "train_twins": count_train_twins(frames),
        "scaler": {
            "type": type(scaler).__name__,
            "fitted_on": "train_benign",
            "flows": int(np.max(scaler.n_samples_seen_)),
            "used_by": "referencia en escala física (el autoencoder ajusta el suyo)",
        },
        "artifacts": {
            name: {"file": paths[name].name, "sha256": sha256_file(paths[name])}
            for name in (*SPLITS, "scaler")
        },
    }
    # Último: su presencia, con hashes válidos, marca un ETL completo.
    atomic_write(
        paths["metadata"],
        lambda tmp: tmp.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        ),
    )
    return paths


def _check_frame(frame: pd.DataFrame) -> None:
    """Invariantes de un split procesado."""
    if forbidden := find_leakage_columns(frame.columns):
        raise ArtifactError(f"Columnas con fuga: {forbidden}")
    if internal := [c for c in INTERNAL_COLUMNS if c in frame.columns]:
        raise ArtifactError(f"Metadata interna sin podar: {internal}")
    features = frame[get_feature_columns(frame.columns)]
    if not (features.dtypes == np.float32).all():
        raise ArtifactError("Hay features que no son float32")
    if not all(np.isfinite(features[c].to_numpy()).all() for c in features):
        raise ArtifactError("Hay features con NaN o Inf")
    benign = (frame[ATTACK_FAMILY_COLUMN] == BENIGN_LABEL).to_numpy()
    if not np.array_equal(benign, frame[TARGET_COLUMN].to_numpy() == 0):
        raise ArtifactError(f"{TARGET_COLUMN} no coincide con la familia")


def load_artifacts(
    dataset: str = DEFAULT_DATASET,
    output_dir: Path | None = None,
    splits: Iterable[str] = SPLITS,
) -> EtlArtifacts:
    """Carga los artefactos y verifica su consistencia.

    Raises:
        ArtifactError: Si falta un archivo, el esquema es de otra versión, un
            hash no coincide o se viola alguna invariante.
    """
    validate_dataset(dataset)
    paths = _artifact_paths(dataset, output_dir)
    if not paths["metadata"].is_file():
        raise ArtifactError(
            f"No existe {paths['metadata']}. Ejecutá: "
            f"python -m src.etl --dataset {dataset}"
        )
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    if metadata["schema_version"] != SCHEMA_VERSION:
        raise ArtifactError(
            f"Esquema {metadata['schema_version']}, se esperaba {SCHEMA_VERSION}. "
            "Reejecutá el ETL."
        )

    splits = tuple(splits)
    # El hash se verifica antes de deserializar.
    for name in (*splits, "scaler"):
        if not paths[name].is_file():
            raise ArtifactError(f"Falta {paths[name]}")
        if sha256_file(paths[name]) != metadata["artifacts"][name]["sha256"]:
            raise ArtifactError(f"{paths[name].name} no coincide con la metadata")

    features = metadata["columns"]["features"]
    columns = [*features, *metadata["columns"]["metadata"]]
    frames = {}
    for name in splits:
        frame = pd.read_parquet(paths[name])
        if list(frame.columns) != columns:
            raise ArtifactError(f"{name}: columnas distintas a las registradas")
        if len(frame) != metadata["counts"][name]["total"]:
            raise ArtifactError(f"{name}: cantidad de flujos inesperada")
        _check_frame(frame)
        frames[name] = frame

    scaler = joblib.load(paths["scaler"])
    if list(scaler.feature_names_in_) != features:
        raise ArtifactError("El scaler se ajustó con otras features")
    return EtlArtifacts(frames, scaler, metadata, features)


def main() -> None:
    parser = argparse.ArgumentParser(description="ETL de flujos de red")
    parser.add_argument(
        "--dataset", choices=SUPPORTED_DATASETS, default=DEFAULT_DATASET
    )
    parser.add_argument(
        "--raw-dir", type=Path, help="CSV ya descargados (no descarga nada)"
    )
    parser.add_argument(
        "--kaggle-dataset", help="mirror de Kaggle (usuario/dataset) vía kagglehub"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        paths = run_etl(args.dataset, args.raw_dir, args.kaggle_dataset)
    except RawDataError as error:
        parser.exit(1, f"error: {error}\n")
    for name, path in paths.items():
        print(f"{name:<9} {path}")


if __name__ == "__main__":
    main()
