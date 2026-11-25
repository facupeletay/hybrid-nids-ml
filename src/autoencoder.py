"""Capa no supervisada: Autoencoder en PyTorch entrenado solo con benignos.

Teoría y decisiones: `docs/theory/03_autoencoder_zero_day.md`.

Uso:
    python -m src.autoencoder --dataset cicids2017
    python -m src.autoencoder --dataset cicids2017 --drop-dst-port --tag prueba

Ni el preprocesador, ni los pesos, ni la época elegida, ni los umbrales dependen
de un flujo de ataque. Los ataques de Val y Test solo se miden al final.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import logging
import math
import os
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.common import (
    NO_PORT_SUFFIX,
    calibrate_thresholds,
    in_dir,
    normalize_tag,
    positive_float,
    positive_int,
    select_features,
    to_strict_json,
    write_report,
)
from src.config import (
    ATTACK_FAMILY_COLUMN,
    ATTACK_LABEL_COLUMN,
    BENIGN_LABEL,
    DEFAULT_DATASET,
    FOCUS_ATTACKS,
    RANDOM_SEED,
    SPLITS,
    SUPPORTED_DATASETS,
    TARGET_COLUMN,
    ensure_directories,
    get_autoencoder_checkpoint_path,
    get_autoencoder_report_path,
)
from src.etl import ArtifactError, atomic_write, load_artifacts, sha256_file
from src.metrics import compute_binary_soc_metrics, compute_per_attack_recall

logger = logging.getLogger(__name__)

CHECKPOINT_VERSION = "1.0.0"
REPORT_SCHEMA_VERSION = "1.0.0"

LOSSES = ("mse", "smooth_l1")
SCORE_METRICS = ("mse", "mae", "huber")
DEVICES = ("auto", "cpu", "cuda")

HIDDEN_DIMS = (64, 32)
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0
LR_FACTOR = 0.5
LR_PATIENCE = 3

# Filas por bloque al transformar y puntuar. Acota la memoria, no el resultado.
CHUNK_ROWS = 65_536


@dataclass(frozen=True)
class AutoencoderParams:
    """Hiperparámetros. Son fijos: no se buscan, y nunca mirando ataques."""

    epochs: int = 50
    batch_size: int = 1024
    lr: float = 1e-3
    patience: int = 8
    bottleneck_dim: int = 12
    dropout: float = 0.1
    clip_z: float = 10.0
    loss: str = "mse"
    score_metric: str = "mse"
    device: str = "auto"
    random_state: int = RANDOM_SEED

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size", "patience", "bottleneck_dim"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} debe ser >= 1")
        for name in ("lr", "clip_z"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"{name} debe ser un número > 0")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout debe estar en [0, 1)")
        for name, options in (
            ("loss", LOSSES),
            ("score_metric", SCORE_METRICS),
            ("device", DEVICES),
        ):
            if getattr(self, name) not in options:
                raise ValueError(
                    f"{name} inválido: {getattr(self, name)!r}. Opciones: {options}"
                )


# --------------------------------------------------------------------------
# Entorno
# --------------------------------------------------------------------------
def resolve_device(device: str) -> str:
    """Traduce `auto`. Pedir `cuda` sin GPU es un error."""
    if device not in DEVICES:
        raise ValueError(f"Device inválido: {device!r}. Opciones: {DEVICES}")
    if device == "cpu":
        return "cpu"
    available = torch.cuda.is_available()
    if device == "cuda" and not available:
        raise ValueError("Se pidió cuda, pero no hay una GPU CUDA utilizable")
    return "cuda" if available else "cpu"


def seed_everything(seed: int) -> None:
    """Fija las semillas y pide kernels deterministas (también en CUDA)."""
    # Lo exige cuBLAS, y debe existir antes de la primera operación en la GPU.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # incluye todas las GPU
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


# --------------------------------------------------------------------------
# Preprocesamiento
# --------------------------------------------------------------------------
def benign_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Única puerta de entrada de datos al ajuste: descarta todos los ataques."""
    benign = frame[frame[TARGET_COLUMN] == 0]
    if benign.empty:
        raise ValueError("No hay tráfico benigno para ajustar el Autoencoder")
    return benign


def signed_log1p(x: np.ndarray) -> np.ndarray:
    """`sign(x) * log(1 + |x|)`: comprime las colas y conserva signo y orden."""
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.log1p(np.abs(x))


@dataclass
class FlowAEPreprocessor:
    """`signed_log1p`, estandarización y recorte, ajustados con benignos de Train."""

    feature_cols: list[str]
    mean: np.ndarray
    scale: np.ndarray
    clip_z: float

    @classmethod
    def fit(
        cls, train: pd.DataFrame, feature_cols: list[str], clip_z: float
    ) -> FlowAEPreprocessor:
        """Ajusta media y escala con las filas `is_attack == 0` de Train."""
        benign = benign_rows(train)
        scaler = StandardScaler()
        for start in range(0, len(benign), CHUNK_ROWS):
            chunk = benign[feature_cols].iloc[start : start + CHUNK_ROWS]
            scaler.partial_fit(signed_log1p(chunk.to_numpy()))
        return cls(list(feature_cols), scaler.mean_, scaler.scale_, float(clip_z))

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Matriz `float32` en `[-clip_z, clip_z]`, con una fila por flujo.

        Raises:
            ValueError: Si faltan columnas o hay valores no finitos. Recortarlos
                los haría pasar por valores extremos legítimos.
        """
        if missing := [c for c in self.feature_cols if c not in frame.columns]:
            raise ValueError(f"Faltan features: {missing}")
        columns = frame[self.feature_cols]
        out = np.empty(columns.shape, dtype=np.float32)
        for start in range(0, len(columns), CHUNK_ROWS):
            rows = slice(start, start + CHUNK_ROWS)
            values = columns.iloc[rows].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError("Hay features con NaN o Inf")
            z = (signed_log1p(values) - self.mean) / self.scale
            out[rows] = np.clip(z, -self.clip_z, self.clip_z)
        return out

    def state_dict(self) -> dict[str, Any]:
        """Solo tensores y tipos básicos, cargables con `weights_only=True`."""
        return {
            "feature_cols": list(self.feature_cols),
            "mean": torch.from_numpy(np.array(self.mean, dtype=np.float64)),
            "scale": torch.from_numpy(np.array(self.scale, dtype=np.float64)),
            "clip_z": float(self.clip_z),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> FlowAEPreprocessor:
        mean, scale = state["mean"].numpy(), state["scale"].numpy()
        features = list(state["feature_cols"])
        if not mean.shape == scale.shape == (len(features),):
            raise ArtifactError("El preprocesador no coincide con sus features")
        return cls(features, mean, scale, float(state["clip_z"]))


# --------------------------------------------------------------------------
# Modelo
# --------------------------------------------------------------------------
def _block(n_in: int, n_out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(n_in, n_out), nn.LayerNorm(n_out), nn.SiLU(), nn.Dropout(dropout)
    )


class FlowAutoencoder(nn.Module):
    """Autoencoder simétrico: `input -> 64 -> 32 -> bottleneck -> 32 -> 64 -> input`.

    El cuello de botella y la salida son lineales. Normalizar o apagar al azar
    unidades del código latente le quitaría capacidad justo donde es escasa.
    """

    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int = 12,
        dropout: float = 0.1,
        hidden_dims: tuple[int, ...] = HIDDEN_DIMS,
    ) -> None:
        super().__init__()
        self.config = {
            "input_dim": int(input_dim),
            "bottleneck_dim": int(bottleneck_dim),
            "dropout": float(dropout),
            "hidden_dims": [int(d) for d in hidden_dims],
        }
        down = [input_dim, *hidden_dims]
        up = [bottleneck_dim, *reversed(hidden_dims)]
        self.encoder = nn.Sequential(
            *(_block(a, b, dropout) for a, b in zip(down, down[1:])),
            nn.Linear(down[-1], bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            *(_block(a, b, dropout) for a, b in zip(up, up[1:])),
            nn.Linear(up[-1], input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def _make_loss(name: str) -> nn.Module:
    return nn.MSELoss() if name == "mse" else nn.SmoothL1Loss()


def _row_error(x_hat: torch.Tensor, x: torch.Tensor, metric: str) -> torch.Tensor:
    if metric == "mse":
        return (x_hat - x).square().mean(dim=1)
    if metric == "mae":
        return (x_hat - x).abs().mean(dim=1)
    return nn.functional.smooth_l1_loss(x_hat, x, reduction="none").mean(dim=1)


def reconstruction_error(
    model: FlowAutoencoder,
    flows: np.ndarray,
    metric: str = "mse",
    device: str = "cpu",
) -> np.ndarray:
    """Error de reconstrucción por flujo (mayor = más anómalo), en `float64`.

    La red se evalúa entera en `float64`, sobre una copia: así la partición en
    bloques y el dispositivo solo alteran el redondeo final.
    """
    if metric not in SCORE_METRICS:
        raise ValueError(f"Métrica inválida: {metric!r}. Opciones: {SCORE_METRICS}")
    network = copy.deepcopy(model).to(device=device, dtype=torch.float64).eval()
    scores = np.empty(len(flows), dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, len(flows), CHUNK_ROWS):
            rows = slice(start, start + CHUNK_ROWS)
            x = torch.from_numpy(flows[rows]).to(device=device, dtype=torch.float64)
            scores[rows] = _row_error(network(x), x, metric).cpu().numpy()
    if not np.isfinite(scores).all():
        raise ValueError("El modelo produjo scores no finitos")
    return scores


# --------------------------------------------------------------------------
# Entrenamiento
# --------------------------------------------------------------------------
def _mean_loss(model: nn.Module, data: torch.Tensor, criterion: nn.Module) -> float:
    model.eval()
    total = 0.0
    with torch.inference_mode():
        for batch in data.split(CHUNK_ROWS):
            total += criterion(model(batch), batch).item() * len(batch)
    return total / len(data)


def train_autoencoder(
    benign_train: np.ndarray,
    benign_val: np.ndarray,
    params: AutoencoderParams | None = None,
) -> tuple[FlowAutoencoder, dict[str, Any]]:
    """Entrena con early stopping sobre la reconstrucción de Val benigno.

    Recibe matrices ya filtradas y sin etiquetas: no puede ver ataques.

    Returns:
        El modelo de la mejor época, en CPU y modo evaluación, y los datos del
        ajuste con la curva por época.
    """
    params = params or AutoencoderParams()
    device = resolve_device(params.device)
    seed_everything(params.random_state)

    model = FlowAutoencoder(
        benign_train.shape[1], params.bottleneck_dim, params.dropout
    ).to(device)
    criterion = _make_loss(params.loss)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params.lr, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=LR_FACTOR, patience=LR_PATIENCE
    )
    # Todo el dataset cabe en la GPU (~0.4 GB): sin DataLoader no hay cuello de
    # botella de CPU y el orden depende solo de este generador.
    train, val = (torch.from_numpy(x).to(device) for x in (benign_train, benign_val))
    shuffler = torch.Generator().manual_seed(params.random_state)

    best_loss, best_epoch, best_state = math.inf, 0, None
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(1, params.epochs + 1):
        model.train()
        lr = optimizer.param_groups[0]["lr"]
        total = torch.zeros((), dtype=torch.float64, device=device)
        order = torch.randperm(len(train), generator=shuffler).to(device)
        for rows in order.split(params.batch_size):
            batch = train[rows]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            total += loss.detach().double() * len(rows)

        train_loss = total.item() / len(train)
        val_loss = _mean_loss(model, val, criterion)
        if not (math.isfinite(train_loss) and math.isfinite(val_loss)):
            raise RuntimeError(f"Pérdida no finita en la época {epoch}")
        scheduler.step(val_loss)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr}
        )
        logger.info(
            "Época %d: train %.6f, val benigno %.6f, lr %.1e",
            *(epoch, train_loss, val_loss, lr),
        )
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= params.patience:
            break

    model.load_state_dict(best_state)
    info = {
        "device": device,
        "n_train_benign": len(train),
        "n_val_benign": len(val),
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "stopped_early": len(history) < params.epochs,
        "best_val_loss": best_loss,
        "fit_seconds": round(time.perf_counter() - started, 1),
        "history": history,
    }
    return model.cpu().eval(), info


def fit(
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: list[str],
    params: AutoencoderParams | None = None,
) -> tuple[FlowAutoencoder, FlowAEPreprocessor, dict[str, Any]]:
    """Ajusta preprocesador y modelo usando solo las filas benignas."""
    params = params or AutoencoderParams()
    preprocessor = FlowAEPreprocessor.fit(train, features, params.clip_z)
    model, info = train_autoencoder(
        preprocessor.transform(benign_rows(train)),
        preprocessor.transform(benign_rows(val)),
        params,
    )
    return model, preprocessor, info


# --------------------------------------------------------------------------
# Checkpoint
# --------------------------------------------------------------------------
def build_suffix(tag: str = "", drop_dst_port: bool = False) -> str:
    """Sufijo de los artefactos, con la misma convención que la capa supervisada."""
    return normalize_tag(tag) + (NO_PORT_SUFFIX if drop_dst_port else "")


def save_checkpoint(
    path: Path,
    model: FlowAutoencoder,
    preprocessor: FlowAEPreprocessor,
    score_metric: str,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Guarda todo lo necesario para puntuar, sin depender de otros archivos."""
    payload = {
        "version": CHECKPOINT_VERSION,
        "model_config": model.config,
        "model_state": model.state_dict(),
        "preprocessor": preprocessor.state_dict(),
        "score_metric": score_metric,
        "metadata": to_strict_json(metadata or {}),
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    atomic_write(path, lambda tmp: tmp.write_bytes(buffer.getvalue()))
    return path


def load_checkpoint(
    path: Path | str,
) -> tuple[FlowAutoencoder, FlowAEPreprocessor, dict[str, Any]]:
    """Recarga modelo (CPU, modo evaluación), preprocesador y resto del checkpoint.

    Usa `weights_only=True`: un `.pt` es un pickle y, sin esa restricción,
    abrirlo puede ejecutar código.

    Raises:
        ArtifactError: Si falta, es de otra versión o es inconsistente.
    """
    path = Path(path)
    if not path.is_file():
        raise ArtifactError(f"No existe {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    version = payload.get("version") if isinstance(payload, dict) else None
    if version != CHECKPOINT_VERSION:
        raise ArtifactError(
            f"{path.name}: versión {version}, se esperaba {CHECKPOINT_VERSION}"
        )
    preprocessor = FlowAEPreprocessor.from_state_dict(payload["preprocessor"])
    model = FlowAutoencoder(**payload["model_config"])
    if model.config["input_dim"] != len(preprocessor.feature_cols):
        raise ArtifactError(f"{path.name}: el modelo no coincide con sus features")
    model.load_state_dict(payload["model_state"])
    rest = {key: payload[key] for key in ("version", "score_metric", "metadata")}
    return model.eval(), preprocessor, rest


def score_dataframe(
    checkpoint_path: Path | str, df: pd.DataFrame, device: str = "auto"
) -> np.ndarray:
    """Score de anomalía por fila de `df`, en `float64` y en el mismo orden.

    Solo necesita el checkpoint. Ignora las columnas que no son features, y el
    orden de las que sí: no hace falta que `df` tenga etiquetas.
    """
    device = resolve_device(device)
    model, preprocessor, rest = load_checkpoint(checkpoint_path)
    flows = preprocessor.transform(df)
    return reconstruction_error(model, flows, rest["score_metric"], device)


def load_autoencoder(
    dataset: str = DEFAULT_DATASET,
    suffix: str = "",
    models_dir: Path | None = None,
    metrics_dir: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Checkpoint y reporte (con los umbrales congelados) de una corrida guardada.

    Args:
        suffix: El de `build_suffix`.

    Raises:
        ArtifactError: Si falta un archivo o el checkpoint no es el del reporte.
    """
    report_path = in_dir(get_autoencoder_report_path(dataset, suffix), metrics_dir)
    if not report_path.is_file():
        raise ArtifactError(f"No existe {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    path = in_dir(get_autoencoder_checkpoint_path(dataset, suffix), models_dir)
    if not path.is_file():
        raise ArtifactError(f"Falta {path}")
    if sha256_file(path) != report["checkpoint"]["sha256"]:
        raise ArtifactError(f"{path.name} no coincide con el reporte")
    return path, report


# --------------------------------------------------------------------------
# Evaluación
# --------------------------------------------------------------------------
def rank_attacks(labels: pd.Series, scores: np.ndarray) -> dict[str, dict[str, Any]]:
    """Separación de cada ataque respecto de los benignos, sin umbral.

    `roc_auc` es la probabilidad de que un flujo del ataque puntúe más que uno
    benigno. Por debajo de 0.5, el ataque se reconstruye mejor que lo normal.
    """
    values = labels.to_numpy()
    is_benign = values == BENIGN_LABEL
    benign = scores[is_benign]
    if benign.size == 0:
        return {}
    ranking = {}
    for name in sorted(set(values[~is_benign])):
        attack = scores[values == name]
        y = np.r_[np.zeros(benign.size), np.ones(attack.size)]
        ranking[str(name)] = {
            "n": int(attack.size),
            "roc_auc": float(roc_auc_score(y, np.r_[benign, attack])),
            "median_score": float(np.median(attack)),
        }
    return dict(sorted(ranking.items(), key=lambda kv: kv[1]["roc_auc"]))


def evaluate(
    val: pd.DataFrame,
    test: pd.DataFrame,
    score: Callable[[pd.DataFrame], np.ndarray],
    focus_attacks: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Calibra en los benignos de Val y mide Test con los umbrales congelados."""
    y_val, y_test = (frame[TARGET_COLUMN].to_numpy() for frame in (val, test))
    score_val = score(val)
    thresholds = calibrate_thresholds(y_val, score_val)

    # Desde acá nada vuelve a influir en el modelo ni en los umbrales.
    score_test = score(test)
    budgets = []
    for budget, threshold in thresholds.items():
        alerts = (score_test >= threshold).astype(np.int8)
        metrics = compute_binary_soc_metrics(y_test, score_test, threshold)
        by_column = {
            column: compute_per_attack_recall(test[column].to_numpy(), alerts)
            for column in (ATTACK_FAMILY_COLUMN, ATTACK_LABEL_COLUMN)
        }
        per_label = by_column[ATTACK_LABEL_COLUMN]["per_attack"]
        budgets.append(
            {
                "max_fpr": budget,
                "threshold": threshold,
                "val": compute_binary_soc_metrics(y_val, score_val, threshold),
                "test": metrics,
                "test_fpr_within_budget": bool(metrics["fpr"] <= budget),
                "focus_attacks": {
                    name: per_label[name] for name in focus_attacks if name in per_label
                },
                **{f"by_{column}": result for column, result in by_column.items()},
            }
        )
    quantiles = (0.5, 0.9, 0.99, 0.999, 1.0)
    return {
        "score_dtype": str(score_test.dtype),
        "budgets": budgets,
        "ranking_by_label_raw": rank_attacks(test[ATTACK_LABEL_COLUMN], score_test),
        "benign_score_quantiles": {
            name: {
                str(q): float(value)
                for q, value in zip(quantiles, np.quantile(s[y == 0], quantiles))
            }
            for name, y, s in (("val", y_val, score_val), ("test", y_test, score_test))
        },
    }


# --------------------------------------------------------------------------
# Orquestación
# --------------------------------------------------------------------------
def run(
    dataset: str = DEFAULT_DATASET,
    params: AutoencoderParams | None = None,
    drop_dst_port: bool = False,
    tag: str = "",
    processed_dir: Path | None = None,
    models_dir: Path | None = None,
    metrics_dir: Path | None = None,
) -> dict[str, Any]:
    """Entrena, calibra el umbral en Val, evalúa en Test y persiste artefactos.

    La evaluación usa el checkpoint releído del disco, no el modelo en memoria:
    lo que se informa es lo que consumirá `hybrid_eval`.

    Returns:
        El reporte, tal como se guarda.
    """
    params = params or AutoencoderParams()
    suffix = build_suffix(tag, drop_dst_port)  # valida antes de entrenar
    device = resolve_device(params.device)

    artifacts = load_artifacts(dataset, output_dir=processed_dir)
    train, val, test = (artifacts.splits[name] for name in SPLITS)
    features, dropped = select_features(train.columns, drop_dst_port)
    model, preprocessor, training = fit(train, val, features, params)

    path = save_checkpoint(
        in_dir(get_autoencoder_checkpoint_path(dataset, suffix), models_dir),
        model,
        preprocessor,
        params.score_metric,
        {
            "dataset": dataset,
            "params": asdict(params),
            "dropped_features": dropped,
            "best_epoch": training["best_epoch"],
            "best_val_loss": training["best_val_loss"],
            "etl_config_hash": artifacts.metadata["config_hash"],
        },
    )
    results = evaluate(
        val,
        test,
        lambda frame: score_dataframe(path, frame, device),
        FOCUS_ATTACKS.get(dataset, ()),
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "dataset": dataset,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "etl_config_hash": artifacts.metadata["config_hash"],
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
        },
        "checkpoint": {"file": path.name, "sha256": sha256_file(path)},
        "params": asdict(params),
        "model_config": model.config,
        "training": training,
        "features": features,
        "dropped_features": dropped,
        **results,
    }
    write_report(
        in_dir(get_autoencoder_report_path(dataset, suffix), metrics_dir), report
    )
    return to_strict_json(report)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def format_report(report: Mapping[str, Any]) -> str:
    """Resumen de consola de un reporte ya convertido a JSON estricto."""

    def pct(value: float | None, digits: int = 2) -> str:
        return "—" if value is None else f"{100 * value:.{digits}f}%"

    def auc(value: float | None) -> str:
        return "—" if value is None else f"{value:.4f}"

    training, budgets = report["training"], report["budgets"]
    first = budgets[0]["test"]
    lines = [
        f"Mejor época {training['best_epoch']} de {training['epochs_run']} "
        f"({training['fit_seconds']:.0f} s en {training['device']}) | "
        f"pérdida Val benigno {training['best_val_loss']:.6f}",
        f"ROC-AUC Test {auc(first['roc_auc'])} | PR-AUC Test {auc(first['pr_auc'])}",
        "",
        f"{'FPR obj.':>8} {'umbral':>10} {'FPR Val':>8} {'FPR Test':>9} "
        f"{'Recall':>8} {'Precis.':>8} {'FA/100k':>8}",
    ]
    for row in budgets:
        test = row["test"]
        lines.append(
            f"{pct(row['max_fpr'], 1):>8} {row['threshold']:>10.6f} "
            f"{pct(row['val']['fpr'], 3):>8} {pct(test['fpr'], 3):>9} "
            f"{pct(test['recall']):>8} {pct(test['precision']):>8} "
            f"{test['alerts_per_100k_benign']:>8.1f}"
        )
    heads = " ".join(f"{'R@' + pct(row['max_fpr'], 1):>8}" for row in budgets)
    lines += ["", f"{'ataque (* destacado)':<30} {'n':>6} {'ROC-AUC':>8} {heads}"]
    for name, rank in report["ranking_by_label_raw"].items():
        mark = " *" if name in budgets[0]["focus_attacks"] else ""
        recalls = " ".join(
            f"{pct(row['by_label_raw']['per_attack'][name]['recall']):>8}"
            for row in budgets
        )
        lines.append(
            f"{name + mark:<30} {rank['n']:>6} {rank['roc_auc']:>8.4f} {recalls}"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    defaults = AutoencoderParams()
    parser = argparse.ArgumentParser(description="Autoencoder de tráfico benigno")
    parser.add_argument(
        "--dataset", choices=SUPPORTED_DATASETS, default=DEFAULT_DATASET
    )
    parser.add_argument("--epochs", type=positive_int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=positive_int, default=defaults.batch_size)
    parser.add_argument("--lr", type=positive_float, default=defaults.lr)
    parser.add_argument(
        "--patience",
        type=positive_int,
        default=defaults.patience,
        help="épocas sin mejora en Val benigno antes de detener",
    )
    parser.add_argument(
        "--bottleneck-dim", type=positive_int, default=defaults.bottleneck_dim
    )
    parser.add_argument(
        "--clip-z",
        type=positive_float,
        default=defaults.clip_z,
        help="recorte en el espacio estandarizado",
    )
    parser.add_argument("--loss", choices=LOSSES, default=defaults.loss)
    parser.add_argument(
        "--score-metric", choices=SCORE_METRICS, default=defaults.score_metric
    )
    parser.add_argument("--device", choices=DEVICES, default=defaults.device)
    parser.add_argument(
        "--drop-dst-port", action="store_true", help="entrena sin puerto destino"
    )
    parser.add_argument("--tag", default="", help="sufijo de los artefactos")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        build_suffix(args.tag)
        resolve_device(args.device)
    except ValueError as error:
        parser.error(str(error))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ensure_directories()
    try:
        report = run(
            args.dataset,
            AutoencoderParams(
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                patience=args.patience,
                bottleneck_dim=args.bottleneck_dim,
                clip_z=args.clip_z,
                loss=args.loss,
                score_metric=args.score_metric,
                device=args.device,
            ),
            args.drop_dst_port,
            args.tag,
        )
    except ArtifactError as error:
        parser.exit(1, f"error: {error}\n")
    print(format_report(report))


if __name__ == "__main__":
    main()
