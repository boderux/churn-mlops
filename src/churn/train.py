"""Model training: several candidates, CV hyper-parameter search, MLflow tracking."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, average_precision_score,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from churn.config import ARTIFACT_DIR, RAW_FEATURES, resolve  # noqa: E402
from churn.features import (  # noqa: E402
    ENGINEERED_CAT, ENGINEERED_NUM, build_preprocessor, feature_engineering_step,
)
from churn.model_io import save_bundle  # noqa: E402

log = logging.getLogger(__name__)
CANDIDATE_DIR = ARTIFACT_DIR / "candidate"
THRESHOLD = 0.5


def model_feature_columns(params: dict[str, Any]) -> list[str]:
    drop = params["features"]["drop_from_model"]
    return [c for c in RAW_FEATURES if c not in drop]


def get_search_spaces(pos_weight: float) -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    return {
        "logreg": (
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            {"clf__C": [0.01, 0.1, 0.5, 1, 5, 10]}),
        "random_forest": (
            RandomForestClassifier(class_weight="balanced_subsample", random_state=42, n_jobs=-1),
            {"clf__n_estimators": [150, 300], "clf__max_depth": [6, 10, 16, None],
             "clf__min_samples_leaf": [1, 5, 10]}),
        "xgboost": (
            XGBClassifier(eval_metric="logloss", scale_pos_weight=pos_weight,
                          random_state=42, n_jobs=2, tree_method="hist"),
            {"clf__n_estimators": [100, 200, 300], "clf__max_depth": [2, 3, 4, 6],
             "clf__learning_rate": [0.03, 0.05, 0.1], "clf__subsample": [0.7, 0.9, 1.0],
             "clf__colsample_bytree": [0.7, 0.9, 1.0]}),
    }


def make_pipeline(clf: Any, drop: list[str]) -> Pipeline:
    return Pipeline([
        ("fe", feature_engineering_step()),
        ("prep", build_preprocessor(drop)),
        ("clf", clf),
    ])


def compute_metrics(y_true: np.ndarray, proba: np.ndarray, thr: float = THRESHOLD) -> dict[str, float]:
    pred = (proba >= thr).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "f1": float(f1_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred)),
        "accuracy": float(accuracy_score(y_true, pred)),
    }


def _plots(pipe: Pipeline, X: pd.DataFrame, y: np.ndarray, outdir: Path, name: str) -> list[Path]:
    paths = []
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(pipe, X, y, ax=ax)
    paths.append(outdir / f"{name}_roc.png")
    fig.savefig(paths[-1], dpi=120, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_estimator(pipe, X, y, ax=ax, colorbar=False)
    paths.append(outdir / f"{name}_cm.png")
    fig.savefig(paths[-1], dpi=120, bbox_inches="tight")
    plt.close(fig)
    return paths


def _git_sha() -> str:
    if os.getenv("GIT_SHA"):
        return os.environ["GIT_SHA"]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL,
            cwd=resolve("."), text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def train(params: dict[str, Any], quick: bool = False) -> dict[str, Any]:
    """Train all candidates, track with MLflow, store best as *candidate* bundle."""
    tcfg, dcfg = params["training"], params["data"]
    proc = resolve(dcfg["processed_dir"])
    train_df = pd.read_csv(proc / "train.csv", dtype={"SeniorCitizen": str})
    test_df = pd.read_csv(proc / "test.csv", dtype={"SeniorCitizen": str})
    feats = model_feature_columns(params)
    y_tr = (train_df[dcfg["target"]] == "Yes").astype(int).to_numpy()
    y_te = (test_df[dcfg["target"]] == "Yes").astype(int).to_numpy()
    X_tr_model, X_te_model = train_df[feats], test_df[feats]
    pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    drop = params["features"]["drop_from_model"]

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"file:{ARTIFACT_DIR / 'mlruns'}"))
    mlflow.set_experiment(tcfg["experiment_name"])
    cv = StratifiedKFold(tcfg["cv_folds"] if not quick else 3, shuffle=True, random_state=42)
    n_iter = tcfg["n_iter"] if not quick else 2
    work = ARTIFACT_DIR / "tmp_plots"
    work.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, Any]] = {}
    best_name, best_cv, best_pipe = "", -1.0, None
    with mlflow.start_run(run_name=f"train-{time.strftime('%Y%m%d-%H%M%S')}") as parent:
        mlflow.set_tags({"git_sha": _git_sha(), "stage": "training"})
        mlflow.log_params({"n_train": len(y_tr), "n_test": len(y_te),
                           "churn_rate": round(float(y_tr.mean()), 4),
                           "dropped_features": ",".join(drop)})
        for name, (clf, space) in get_search_spaces(pos_weight).items():
            if name not in tcfg["models"]:
                continue
            with mlflow.start_run(run_name=name, nested=True):
                search = RandomizedSearchCV(
                    make_pipeline(clf, drop), space, n_iter=min(n_iter, int(np.prod(
                        [len(v) for v in space.values()]))), cv=cv,
                    scoring=tcfg["scoring"], n_jobs=1, random_state=42, refit=True)
                t0 = time.time()
                search.fit(X_tr_model, y_tr)
                fit_s = time.time() - t0
                proba = search.predict_proba(X_te_model)[:, 1]
                m = compute_metrics(y_te, proba)
                m["cv_roc_auc"] = float(search.best_score_)
                mlflow.set_tag("model_type", name)
                mlflow.log_params({k.replace("clf__", ""): v for k, v in search.best_params_.items()})
                mlflow.log_metrics({f"test_{k}" if k != "cv_roc_auc" else k: v for k, v in m.items()})
                mlflow.log_metric("fit_seconds", fit_s)
                for p in _plots(search.best_estimator_, X_te_model, y_te, work, name):
                    mlflow.log_artifact(str(p), "plots")
                mlflow.sklearn.log_model(search.best_estimator_, "model")
                results[name] = {"metrics": m, "params": search.best_params_}
                log.info("%s: %s", name, {k: round(v, 4) for k, v in m.items()})
                if m["cv_roc_auc"] > best_cv:
                    best_name, best_cv, best_pipe = name, m["cv_roc_auc"], search.best_estimator_
        assert best_pipe is not None, "no model trained"
        best = results[best_name]
        mlflow.set_tags({"best_model": best_name})
        mlflow.log_metrics({f"best_{k}": v for k, v in best["metrics"].items()})

    reference = train_df[feats].sample(min(2000, len(train_df)), random_state=42).reset_index(drop=True)
    meta = {
        "version": time.strftime("%Y%m%d%H%M%S"),
        "model_type": best_name, "trained_at": time.time(), "git_sha": _git_sha(),
        "mlflow_run_id": parent.info.run_id, "metrics": best["metrics"],
        "params": {k: str(v) for k, v in best["params"].items()},
        "all_candidates": {k: v["metrics"] for k, v in results.items()},
        "feature_columns": feats, "threshold": THRESHOLD,
        "engineered": ENGINEERED_NUM + ENGINEERED_CAT,
    }
    save_bundle(CANDIDATE_DIR, best_pipe, meta, reference)
    (ARTIFACT_DIR / "candidate_metrics.json").write_text(json.dumps(meta, indent=2, default=str))
    _try_register(best_pipe, tcfg["registered_model_name"], meta)
    return meta


def _try_register(pipe: Pipeline, name: str, meta: dict[str, Any]) -> None:
    """Register the winning model in the MLflow Model Registry (best effort)."""
    try:
        with mlflow.start_run(run_id=meta["mlflow_run_id"]):
            info = mlflow.sklearn.log_model(pipe, "champion_candidate", registered_model_name=name)
            log.info("Registered model: %s", info.model_uri)
    except Exception as exc:  # noqa: BLE001
        log.warning("Model registry unavailable: %s", exc)
