"""Capa supervisada: XGBoost / LightGBM binario, calibrado por presupuesto de FPR.

Teoría y decisiones: `docs/theory/02_supervised_soc_baseline.md`.

Uso:
    python -m src.baseline_xgb --dataset cicids2017
    python -m src.baseline_xgb --dataset cicids2017 --run-ablation
    python -m src.baseline_xgb --dataset cicids2017 --model lgbm --tag prueba

Train ajusta, Val decide (early stopping y umbrales) y Test solo se mide. Los
árboles consumen los parquets sin escalar.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import subprocess
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import expit

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
    RANDOM_SEED,
    SPLITS,
    SUPPORTED_DATASETS,
    TARGET_COLUMN,
    ensure_directories,
    get_baseline_model_path,
    get_baseline_report_path,
)
from src.etl import ArtifactError, atomic_write, load_artifacts, sha256_file
from src.metrics import (
    compute_binary_soc_metrics,
    compute_per_attack_recall,
)

logger = logging.getLogger(__name__)

REPORT_SCHEMA_VERSION = "1.0.0"

MODELS = ("xgb", "lgbm")
MODEL_EXTENSIONS = {"xgb": "json", "lgbm": "txt"}
DEVICES = ("auto", "cpu", "cuda")
BALANCE_MODES = ("train", "none")

EARLY_STOPPING_ROUNDS = 50
TOP_FEATURES = 20
# LightGBM crece por hojas: sin tope, una profundidad alta daría 2^d - 1 hojas.
MAX_LGBM_LEAVES = 255

LGBM_SUFFIX = "_lgbm"


@dataclass(frozen=True)
class TrainParams:
    """Hiperparámetros. Son fijos: no se buscan, y nunca mirando Test.

    Attributes:
        balance: `"train"` (benignos / ataques de Train), `"none"` o un peso
            positivo para la clase ataque.
        device: `"auto"` usa CUDA si está disponible (solo XGBoost).
    """

    model: str = "xgb"
    n_estimators: int = 1000
    max_depth: int = 8
    learning_rate: float = 0.1
    balance: str | float = "train"
    device: str = "auto"
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS
    random_state: int = RANDOM_SEED

    def __post_init__(self) -> None:
        if self.model not in MODELS:
            raise ValueError(f"Modelo inválido: {self.model!r}. Opciones: {MODELS}")
        if self.device not in DEVICES:
            raise ValueError(f"Device inválido: {self.device!r}. Opciones: {DEVICES}")
        parse_balance(self.balance)


@dataclass
class SupervisedModel:
    """Booster nativo, recortado a su mejor iteración, y el orden de sus features."""

    backend: str
    booster: xgb.Booster | lgb.Booster
    features: list[str]


# --------------------------------------------------------------------------
# Configuración del entrenamiento
# --------------------------------------------------------------------------
def parse_balance(value: str | float) -> str | float:
    """Valida `balance`: un modo de `BALANCE_MODES` o un peso positivo y finito."""
    if value in BALANCE_MODES:
        return value
    try:
        weight = float(value)
    except (TypeError, ValueError):
        weight = math.nan
    if not (math.isfinite(weight) and weight > 0):
        raise ValueError(
            f"balance inválido: {value!r}. Opciones: {BALANCE_MODES} o un peso > 0"
        )
    return weight


def compute_scale_pos_weight(y_train: np.ndarray, balance: str | float) -> float:
    """Peso de la clase ataque. Solo recibe las etiquetas de Train."""
    n_attack = int(np.count_nonzero(y_train == 1))
    n_benign = int(np.count_nonzero(y_train == 0))
    if n_attack == 0 or n_benign == 0:
        raise ValueError("Train debe tener flujos benignos y ataques")
    balance = parse_balance(balance)
    if balance == "train":
        return n_benign / n_attack
    return 1.0 if balance == "none" else balance


def cuda_available() -> bool:
    """CUDA utilizable por XGBoost: build con soporte y al menos una GPU visible.

    Se consulta a `nvidia-smi` porque, sin GPU, XGBoost no falla: advierte y
    entrena en CPU.
    """
    executable = shutil.which("nvidia-smi")
    if executable is None or not xgb.build_info().get("USE_CUDA", False):
        return False
    try:
        result = subprocess.run(
            [executable, "-L"], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and "GPU" in result.stdout


def resolve_device(device: str, model: str) -> str:
    """Traduce `auto`. Pedir `cuda` donde no se puede usar es un error."""
    if model == "lgbm":
        if device == "cuda":
            raise ValueError("Los wheels de LightGBM no incluyen CUDA: usá cpu o auto")
        return "cpu"
    if device == "cpu":
        return "cpu"
    available = cuda_available()
    if device == "cuda" and not available:
        raise ValueError("Se pidió cuda, pero no hay una GPU CUDA utilizable")
    return "cuda" if available else "cpu"


# --------------------------------------------------------------------------
# Entrenamiento y scores
# --------------------------------------------------------------------------
def _target(frame: pd.DataFrame) -> np.ndarray:
    return frame[TARGET_COLUMN].to_numpy()


def train_supervised(
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: list[str],
    params: TrainParams | None = None,
) -> tuple[SupervisedModel, dict[str, Any]]:
    """Entrena con early stopping por PR-AUC en Val.

    Returns:
        El modelo recortado a su mejor iteración y los datos del ajuste.
    """
    params = params or TrainParams()
    x_train, y_train = train[features], _target(train)
    x_val, y_val = val[features], _target(val)
    weight = compute_scale_pos_weight(y_train, params.balance)
    device = resolve_device(params.device, params.model)
    if y_val.min() == y_val.max():
        raise ValueError("Val debe tener ambas clases para el early stopping")

    common = {
        "n_estimators": params.n_estimators,
        "max_depth": params.max_depth,
        "learning_rate": params.learning_rate,
        "scale_pos_weight": weight,
        "random_state": params.random_state,
        "n_jobs": -1,
    }
    started = time.perf_counter()
    if params.model == "xgb":
        classifier = xgb.XGBClassifier(
            **common,
            objective="binary:logistic",
            tree_method="hist",
            eval_metric="aucpr",
            early_stopping_rounds=params.early_stopping_rounds,
            device=device,
        )
        classifier.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        n_trees = classifier.best_iteration + 1
        val_pr_auc = classifier.best_score
        # Solo los árboles que se usan: si no, la importancia contaría el resto.
        booster = classifier.get_booster()[:n_trees]
        booster.set_param({"device": "cpu"})
    else:
        classifier = lgb.LGBMClassifier(
            **common,
            objective="binary",
            metric="average_precision",
            num_leaves=min(2**params.max_depth - 1, MAX_LGBM_LEAVES),
            deterministic=True,
            force_col_wise=True,
            verbose=-1,
        )
        classifier.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            callbacks=[lgb.early_stopping(params.early_stopping_rounds, verbose=False)],
        )
        n_trees = classifier.best_iteration_ or params.n_estimators
        val_pr_auc = classifier.best_score_["valid_0"]["average_precision"]
        text = classifier.booster_.model_to_string(num_iteration=n_trees)
        booster = lgb.Booster(model_str=text)

    info = {
        "device": device,
        "scale_pos_weight": weight,
        "n_trees": int(n_trees),
        "stopped_early": bool(n_trees < params.n_estimators),
        "early_stopping_val_pr_auc": float(val_pr_auc),
        "fit_seconds": round(time.perf_counter() - started, 1),
    }
    return SupervisedModel(params.model, booster, list(features)), info


def score_supervised(model: SupervisedModel, flows: pd.DataFrame) -> np.ndarray:
    """`P(ataque)` por flujo, en `float64`.

    Con XGBoost se toma el margen y la sigmoide se aplica en `float64`: en
    `float32` los scores altos se saturan y empatan, justo donde cae el umbral.
    """
    x = flows[model.features]
    if model.backend == "xgb":
        margin = model.booster.inplace_predict(x, predict_type="margin")
        scores = expit(np.asarray(margin, dtype=np.float64))
    else:
        scores = np.asarray(model.booster.predict(x), dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("El modelo produjo scores no finitos")
    return scores


def top_features(
    model: SupervisedModel, limit: int = TOP_FEATURES
) -> list[dict[str, Any]]:
    """Features con mayor `gain` medio por división, con su ganancia total."""
    if model.backend == "xgb":
        total = model.booster.get_score(importance_type="total_gain")
        splits = model.booster.get_score(importance_type="weight")
    else:
        # LightGBM reescribe los nombres con espacios: se alinea por posición.
        gains = model.booster.feature_importance("gain")
        counts = model.booster.feature_importance("split")
        total = dict(zip(model.features, gains))
        splits = dict(zip(model.features, counts))

    grand_total = float(sum(total.values()))
    rows = [
        {
            "feature": name,
            "gain": float(total[name]) / float(splits[name]),
            "total_gain": float(total[name]),
            "total_gain_share": float(total[name]) / grand_total,
            "splits": int(splits[name]),
        }
        for name in model.features
        if splits.get(name, 0) and grand_total
    ]
    return sorted(rows, key=lambda row: row["gain"], reverse=True)[:limit]


# --------------------------------------------------------------------------
# Gemelos
# --------------------------------------------------------------------------
def _row_bytes(frame: pd.DataFrame, features: list[str]) -> list[bytes]:
    # `+ 0.0` unifica -0.0 y 0.0, iguales para el modelo y distintos en bytes.
    matrix = np.ascontiguousarray(frame[features].to_numpy() + 0.0)
    return [row.tobytes() for row in matrix]


def find_train_twins(
    train: pd.DataFrame, flows: pd.DataFrame, features: list[str]
) -> np.ndarray:
    """Máscara de `flows` cuyo vector **exacto** de features existe en Train.

    El hash de 64 bits solo preselecciona candidatos; la igualdad se confirma
    byte a byte, así que una colisión no puede producir un gemelo falso.
    """

    def hashes(frame: pd.DataFrame) -> np.ndarray:
        clean = frame[features] + 0.0
        return pd.util.hash_pandas_object(clean, index=False).to_numpy()

    train_hash, flow_hash = hashes(train), hashes(flows)
    candidates = np.flatnonzero(np.isin(flow_hash, train_hash))
    twins = np.zeros(len(flows), dtype=bool)
    if candidates.size == 0:
        return twins
    sources = np.flatnonzero(np.isin(train_hash, flow_hash[candidates]))
    known = set(_row_bytes(train.iloc[sources], features))
    twins[candidates] = [
        row in known for row in _row_bytes(flows.iloc[candidates], features)
    ]
    return twins


def summarize_twins(labels: pd.Series, twins: np.ndarray) -> dict[str, dict[str, Any]]:
    """Cantidad y fracción de gemelos por etiqueta, benignos incluidos."""
    grouped = pd.Series(twins).groupby(labels.to_numpy())
    return {
        str(label): {
            "n": int(rows.size),
            "twins": int(rows.sum()),
            "twin_fraction": float(rows.mean()),
        }
        for label, rows in grouped
    }


# --------------------------------------------------------------------------
# Calibración y evaluación
# --------------------------------------------------------------------------
def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else math.nan


def breakdown_by_attack(
    labels: pd.Series,
    alerts: np.ndarray,
    twins: np.ndarray,
    train_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Recall por ataque, con y sin gemelos, separando lo ausente de Train.

    Returns:
        - `seen`: ataques con muestras en Train.
        - `unseen`: ataques sin ninguna. Su recall mide generalización y suele
          apoyarse en pocos flujos.
        - `macro_recall` (todos), `macro_recall_seen` y
          `macro_recall_seen_novel` (vistos, sin gemelos).
        - `benign`: FPR sobre todos los benignos y sin gemelos.
    """
    values = labels.to_numpy()
    full = compute_per_attack_recall(values, alerts)

    def subset(mask: np.ndarray) -> dict[str, Any]:
        if not mask.any():
            return {"per_attack": {}, "benign": {"n": 0, "fpr": math.nan}}
        return compute_per_attack_recall(values[mask], alerts[mask])

    novel, twin = subset(~twins), subset(twins)
    empty = {"n": 0, "recall": math.nan}

    seen, unseen = {}, {}
    for name, row in full["per_attack"].items():
        novel_row = novel["per_attack"].get(name, empty)
        twin_row = twin["per_attack"].get(name, empty)
        n_train = int(train_counts.get(name, 0))
        (seen if n_train else unseen)[name] = {
            **row,
            "n_train": n_train,
            "n_twins": twin_row["n"],
            "recall_twins": twin_row["recall"],
            "n_novel": novel_row["n"],
            "recall_novel": novel_row["recall"],
            "twin_inflation": row["recall"] - novel_row["recall"],
        }

    return {
        "seen": seen,
        "unseen": unseen,
        "macro_recall": full["macro_recall"],
        "macro_recall_seen": _mean(r["recall"] for r in seen.values()),
        "macro_recall_seen_novel": _mean(
            r["recall_novel"] for r in seen.values() if r["n_novel"]
        ),
        "benign": {
            **full["benign"],
            "n_novel": novel["benign"]["n"],
            "fpr_novel": novel["benign"]["fpr"],
        },
    }


def evaluate_test(
    test: pd.DataFrame,
    scores: np.ndarray,
    thresholds: Mapping[float, float],
    twins: np.ndarray,
    train_counts: Mapping[str, Mapping[str, int]],
) -> list[dict[str, Any]]:
    """Mide Test con umbrales ya congelados, uno por presupuesto."""
    y, novel = _target(test), ~twins
    rows = []
    for budget, threshold in thresholds.items():
        alerts = (scores >= threshold).astype(np.int8)
        metrics = compute_binary_soc_metrics(y, scores, threshold)
        rows.append(
            {
                "max_fpr": budget,
                "threshold": threshold,
                "test": metrics,
                "test_fpr_within_budget": bool(metrics["fpr"] <= budget),
                "test_excluding_twins": (
                    compute_binary_soc_metrics(y[novel], scores[novel], threshold)
                    if novel.any()
                    else None
                ),
                **{
                    f"by_{column}": breakdown_by_attack(
                        test[column], alerts, twins, train_counts[column]
                    )
                    for column in (ATTACK_FAMILY_COLUMN, ATTACK_LABEL_COLUMN)
                },
            }
        )
    return rows


def fit_and_evaluate(
    splits: Mapping[str, pd.DataFrame],
    params: TrainParams | None = None,
    drop_dst_port: bool = False,
) -> tuple[SupervisedModel, dict[str, Any]]:
    """Entrena en Train, calibra en Val y mide en Test. No toca el disco."""
    params = params or TrainParams()
    train, val, test = (splits[name] for name in SPLITS)
    features, dropped = select_features(train.columns, drop_dst_port)

    model, training = train_supervised(train, val, features, params)
    logger.info(
        "%s: %d árboles en %s, %.1f s, PR-AUC en Val %.6f",
        params.model,
        training["n_trees"],
        training["device"],
        training["fit_seconds"],
        training["early_stopping_val_pr_auc"],
    )

    score_val = score_supervised(model, val)
    thresholds = calibrate_thresholds(_target(val), score_val)

    # Desde acá nada vuelve a influir en el modelo ni en los umbrales.
    score_test = score_supervised(model, test)
    twins = find_train_twins(train, test, features)
    train_counts = {
        column: train[column].value_counts().to_dict()
        for column in (ATTACK_FAMILY_COLUMN, ATTACK_LABEL_COLUMN)
    }
    budgets = evaluate_test(test, score_test, thresholds, twins, train_counts)
    for row in budgets:
        row["val"] = compute_binary_soc_metrics(
            _target(val), score_val, row["threshold"]
        )

    report = {
        "params": asdict(params),
        "training": training,
        "features": features,
        "dropped_features": dropped,
        "score_dtype": str(score_test.dtype),
        "unseen_in_train": {
            column: sorted(set(test[column]) - {BENIGN_LABEL} - set(counts))
            for column, counts in train_counts.items()
        },
        "budgets": budgets,
        "twin_audit": {
            "definition": "vector exacto de las features del modelo, en Train",
            "n_features": len(features),
            "by_label": summarize_twins(test[ATTACK_LABEL_COLUMN], twins),
        },
        "top_features_by_gain": top_features(model),
    }
    return model, report


def compare_ablation(
    with_port: Mapping[str, Any], without_port: Mapping[str, Any]
) -> dict[str, Any]:
    """Impacto de quitar el puerto destino (`delta` = sin puerto - con puerto).

    El recall es el del presupuesto más estricto, cada modelo con su propio
    umbral de Val. Sin el puerto hay más gemelos, así que las métricas que los
    excluyen se miden sobre poblaciones distintas: se informan sin `delta`.
    """

    def strictest(report: Mapping[str, Any]) -> Mapping[str, Any]:
        return min(report["budgets"], key=lambda item: item["max_fpr"])

    def summary(report: Mapping[str, Any]) -> dict[str, float]:
        row = strictest(report)
        by_label = row[f"by_{ATTACK_LABEL_COLUMN}"]
        return {
            "test_pr_auc": row["test"]["pr_auc"],
            "test_fpr": row["test"]["fpr"],
            "recall": row["test"]["recall"],
            "macro_recall_seen": by_label["macro_recall_seen"],
        }

    def excluding_twins(report: Mapping[str, Any]) -> dict[str, float]:
        row = strictest(report)
        novel = row["test_excluding_twins"] or {}
        groups = row[f"by_{ATTACK_LABEL_COLUMN}"]
        return {
            "n_attack": novel.get("n_attack", 0),
            "pr_auc": novel.get("pr_auc", math.nan),
            "recall": novel.get("recall", math.nan),
            "macro_recall_seen": groups["macro_recall_seen_novel"],
        }

    def twin_fractions(report: Mapping[str, Any]) -> dict[str, float]:
        audit = report["twin_audit"]["by_label"]
        return {name: row["twin_fraction"] for name, row in audit.items()}

    def recalls(report: Mapping[str, Any]) -> dict[str, float]:
        groups = strictest(report)[f"by_{ATTACK_LABEL_COLUMN}"]
        merged = {**groups["seen"], **groups["unseen"]}
        return {name: row["recall"] for name, row in merged.items()}

    def deltas(
        base: Mapping[str, float], ablated: Mapping[str, float]
    ) -> dict[str, dict[str, float]]:
        return {
            name: {
                "with_port": base[name],
                "without_port": ablated[name],
                "delta": ablated[name] - base[name],
            }
            for name in base
        }

    return {
        "dropped_features": list(without_port["dropped_features"]),
        "max_fpr": strictest(with_port)["max_fpr"],
        "summary": deltas(summary(with_port), summary(without_port)),
        "recall_by_label": deltas(recalls(with_port), recalls(without_port)),
        "twin_fraction_by_label": deltas(
            twin_fractions(with_port), twin_fractions(without_port)
        ),
        "excluding_own_twins": {
            "with_port": excluding_twins(with_port),
            "without_port": excluding_twins(without_port),
        },
    }


# --------------------------------------------------------------------------
# Persistencia
# --------------------------------------------------------------------------
def build_suffix(tag: str = "", model: str = "xgb", drop_dst_port: bool = False) -> str:
    """Sufijo de los artefactos. Cada variante tiene el suyo y no pisa a las demás."""
    variant = LGBM_SUFFIX if model == "lgbm" else ""
    return normalize_tag(tag) + variant + (NO_PORT_SUFFIX if drop_dst_port else "")


def save_model(model: SupervisedModel, path: Path) -> Path:
    """Guarda el modelo en el formato de texto nativo del backend."""
    if model.backend == "xgb":
        payload = bytes(model.booster.save_raw(raw_format="json"))
    else:
        payload = model.booster.model_to_string().encode("utf-8")
    # En bytes: en modo texto, Windows escribe CRLF y LightGBM no puede releerlo.
    atomic_write(path, lambda tmp: tmp.write_bytes(payload))
    return path


def load_model(path: Path, features: list[str]) -> SupervisedModel:
    """Recarga un modelo; el backend se deduce de la extensión."""
    if path.suffix == ".json":
        booster = xgb.Booster(model_file=bytearray(path.read_bytes()))
        if booster.feature_names != features:
            raise ArtifactError(f"{path.name} se entrenó con otras features")
        return SupervisedModel("xgb", booster, list(features))
    booster = lgb.Booster(model_str=path.read_bytes().decode("utf-8"))
    if booster.num_feature() != len(features):
        raise ArtifactError(f"{path.name} se entrenó con otras features")
    return SupervisedModel("lgbm", booster, list(features))


def load_baseline(
    dataset: str = DEFAULT_DATASET,
    suffix: str = "",
    models_dir: Path | None = None,
    metrics_dir: Path | None = None,
) -> tuple[SupervisedModel, dict[str, Any]]:
    """Modelo y reporte (con los umbrales congelados) de una corrida guardada.

    Args:
        suffix: El de `build_suffix`.

    Raises:
        ArtifactError: Si falta un archivo o el modelo no es el del reporte.
    """
    report_path = in_dir(get_baseline_report_path(dataset, suffix), metrics_dir)
    if not report_path.is_file():
        raise ArtifactError(f"No existe {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    extension = MODEL_EXTENSIONS[report["params"]["model"]]
    path = in_dir(get_baseline_model_path(dataset, suffix, extension), models_dir)
    if not path.is_file():
        raise ArtifactError(f"Falta {path}")
    if sha256_file(path) != report["model_artifact"]["sha256"]:
        raise ArtifactError(f"{path.name} no coincide con el reporte")
    return load_model(path, report["features"]), report


# --------------------------------------------------------------------------
# Orquestación
# --------------------------------------------------------------------------
def run(
    dataset: str = DEFAULT_DATASET,
    params: TrainParams | None = None,
    drop_dst_port: bool = False,
    run_ablation: bool = False,
    tag: str = "",
    processed_dir: Path | None = None,
    models_dir: Path | None = None,
    metrics_dir: Path | None = None,
) -> dict[str, Any]:
    """Entrena, calibra en Val, evalúa en Test y persiste modelo y reporte.

    Con `run_ablation` repite la corrida sin puerto destino, guarda también sus
    artefactos y agrega la comparativa al reporte principal.

    Returns:
        El reporte principal, tal como se guarda.
    """
    params = params or TrainParams()
    if run_ablation and drop_dst_port:
        raise ValueError("La ablación ya incluye la corrida sin puerto destino")
    build_suffix(tag)  # valida antes de entrenar

    artifacts = load_artifacts(dataset, output_dir=processed_dir)
    splits = artifacts.splits
    if run_ablation:
        select_features(artifacts.feature_columns, drop_dst_port=True)

    def execute(drop: bool) -> tuple[dict[str, Any], Path]:
        model, results = fit_and_evaluate(splits, params, drop)
        suffix = build_suffix(tag, params.model, drop)
        extension = MODEL_EXTENSIONS[params.model]
        model_path = save_model(
            model,
            in_dir(get_baseline_model_path(dataset, suffix, extension), models_dir),
        )
        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "dataset": dataset,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "etl_config_hash": artifacts.metadata["config_hash"],
            "versions": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "xgboost": xgb.__version__,
                "lightgbm": lgb.__version__,
            },
            "model_artifact": {
                "file": model_path.name,
                "sha256": sha256_file(model_path),
            },
            "counts": {
                name: {
                    "total": len(frame),
                    "attacks": int(np.count_nonzero(_target(frame))),
                }
                for name, frame in splits.items()
            },
            **results,
        }
        path = in_dir(get_baseline_report_path(dataset, suffix), metrics_dir)
        write_report(path, report)
        return report, path

    report, path = execute(drop_dst_port)
    if run_ablation:
        without_port, _ = execute(True)
        report["ablation_dst_port"] = compare_ablation(report, without_port)
        write_report(path, report)
    return to_strict_json(report)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _pct(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{100 * value:.{digits}f}%"


def format_report(report: Mapping[str, Any]) -> str:
    """Resumen de consola de un reporte ya convertido a JSON estricto."""
    first = report["budgets"][0]
    lines = [
        f"PR-AUC Test {first['test']['pr_auc']:.6f} | "
        f"ROC-AUC Test {first['test']['roc_auc']:.6f} | "
        f"PR-AUC Val {first['val']['pr_auc']:.6f}",
        "",
        f"{'FPR obj.':>8} {'umbral':>10} {'FPR Val':>8} {'FPR Test':>9} "
        f"{'Recall':>8} {'s/gem.':>8} {'Precis.':>8} {'FA/100k':>8}",
    ]
    for row in report["budgets"]:
        test, novel = row["test"], row["test_excluding_twins"] or {}
        lines.append(
            f"{_pct(row['max_fpr'], 1):>8} {row['threshold']:>10.6f} "
            f"{_pct(row['val']['fpr'], 3):>8} {_pct(test['fpr'], 3):>9} "
            f"{_pct(test['recall']):>8} {_pct(novel.get('recall')):>8} "
            f"{_pct(test['precision']):>8} {test['alerts_per_100k_benign']:>8.1f}"
        )

    lines += [
        "",
        f"Recall en Test @ FPR {_pct(first['max_fpr'], 1)} (* = sin muestras en Train)",
        f"{'ataque':<28} {'n':>6} {'recall':>8} {'gemelos':>8} "
        f"{'n s/gem.':>8} {'recall s/gem.':>13}",
    ]
    groups = first[f"by_{ATTACK_LABEL_COLUMN}"]
    for mark, rows in (("", groups["seen"]), (" *", groups["unseen"])):
        for name, row in rows.items():
            lines.append(
                f"{name + mark:<28} {row['n']:>6} {_pct(row['recall']):>8} "
                f"{row['n_twins']:>8} {row['n_novel']:>8} "
                f"{_pct(row['recall_novel']):>13}"
            )

    if ablation := report.get("ablation_dst_port"):
        lines += [
            "",
            f"Ablación: sin {', '.join(ablation['dropped_features'])}",
            f"{'':<30} {'con':>10} {'sin':>10} {'delta':>10}",
        ]
        for name, row in ablation["summary"].items():
            values = (row["with_port"], row["without_port"], row["delta"])
            cells = ("—" if v is None else f"{v:.5f}" for v in values)
            lines.append(f"{name:<30} " + " ".join(f"{c:>10}" for c in cells))
    return "\n".join(lines)


def _balance(text: str) -> str | float:
    try:
        return parse_balance(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def build_parser() -> argparse.ArgumentParser:
    defaults = TrainParams()
    parser = argparse.ArgumentParser(description="Clasificador supervisado para SOC")
    parser.add_argument(
        "--dataset", choices=SUPPORTED_DATASETS, default=DEFAULT_DATASET
    )
    parser.add_argument("--model", choices=MODELS, default=defaults.model)
    parser.add_argument(
        "--n-estimators", type=positive_int, default=defaults.n_estimators
    )
    parser.add_argument("--max-depth", type=positive_int, default=defaults.max_depth)
    parser.add_argument(
        "--learning-rate", type=positive_float, default=defaults.learning_rate
    )
    parser.add_argument(
        "--balance",
        type=_balance,
        default=defaults.balance,
        help="peso de la clase ataque: train (benignos/ataques), none o un número",
    )
    parser.add_argument("--device", choices=DEVICES, default=defaults.device)
    port = parser.add_mutually_exclusive_group()
    port.add_argument(
        "--drop-dst-port", action="store_true", help="entrena sin puerto destino"
    )
    port.add_argument(
        "--run-ablation",
        action="store_true",
        help="repite la corrida sin puerto destino y guarda la comparativa",
    )
    parser.add_argument("--tag", default="", help="sufijo de los artefactos")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        build_suffix(args.tag)
        resolve_device(args.device, args.model)
    except ValueError as error:
        parser.error(str(error))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ensure_directories()
    try:
        report = run(
            args.dataset,
            TrainParams(
                model=args.model,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                learning_rate=args.learning_rate,
                balance=args.balance,
                device=args.device,
            ),
            args.drop_dst_port,
            args.run_ablation,
            args.tag,
        )
    except ArtifactError as error:
        parser.exit(1, f"error: {error}\n")
    print(format_report(report))


if __name__ == "__main__":
    main()
