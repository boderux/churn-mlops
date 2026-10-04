"""FastAPI serving layer for the churn model.

Endpoints (v1): /v1/predict, /v1/predict/batch, /v1/explain, /v1/model, /v1/admin/reload
Ops: /health (liveness), /ready (readiness), /metrics (Prometheus)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from api import metrics as m
from api.schemas import (
    BatchRequest,
    BatchResponse,
    Customer,
    ErrorResponse,
    Explanation,
    FeatureContribution,
    ModelInfo,
    Prediction,
)
from churn import __version__
from churn.config import ARTIFACT_DIR, CHAMPION_DIR
from churn.explain import ShapExplainer
from churn.model_io import load_bundle, load_reference

log = logging.getLogger("churn.api")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


class ModelStore:
    """Thread-safe holder for the champion model (hot-reloadable)."""

    def __init__(self) -> None:
        self.pipe: Any = None
        self.meta: dict[str, Any] = {}
        self.explainer: ShapExplainer | None = None
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self.pipe is not None

    def load(self, directory: Path | None = None) -> bool:
        directory = directory or Path(os.getenv("MODEL_DIR", CHAMPION_DIR))
        try:
            pipe, meta = load_bundle(directory)
            ref = load_reference(directory)
            explainer = ShapExplainer(pipe, ref) if ref is not None else None
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not load model from %s: %s", directory, exc)
            m.MODEL_LOADED.set(0)
            return False
        with self._lock:
            self.pipe, self.meta, self.explainer = pipe, meta, explainer
        m.MODEL_INFO.clear()
        m.MODEL_INFO.labels(meta["version"], meta["model_type"]).set(1)
        m.MODEL_AUC.set(meta["metrics"]["roc_auc"])
        m.MODEL_RECALL.set(meta["metrics"]["recall"])
        m.MODEL_TRAINED_TS.set(meta["trained_at"])
        m.MODEL_LOADED.set(1)
        log.info("Loaded model %s (%s)", meta["version"], meta["model_type"])
        return True


store = ModelStore()
_log_lock = threading.Lock()


def _prediction_log_path() -> Path:
    return Path(os.getenv("PREDICTION_LOG", ARTIFACT_DIR / "prediction_log.jsonl"))


def log_prediction(features: dict[str, Any], proba: float, request_id: str) -> None:
    """Append to the JSONL log consumed by the Evidently drift DAG."""
    if os.getenv("LOG_PREDICTIONS", "true").lower() != "true":
        return
    rec = {"ts": time.time(), "request_id": request_id, "features": features, "probability": proba}
    path = _prediction_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _log_lock, open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
    except OSError as exc:
        log.error("prediction log write failed: %s", exc)


def risk_level(p: float) -> Literal["low", "medium", "high"]:
    return "low" if p < 0.3 else "medium" if p < 0.6 else "high"


def _require_model() -> ModelStore:
    if not store.ready:
        raise HTTPException(503, "Model not loaded")
    return store


def _predict(customers: list[Customer]) -> list[Prediction]:
    s = _require_model()
    feats = s.meta["feature_columns"]
    rows = [c.model_dump() for c in customers]
    df = pd.DataFrame(rows)[feats]
    try:
        probas = s.pipe.predict_proba(df)[:, 1]
    except Exception as exc:  # noqa: BLE001
        m.PREDICTION_ERRORS.inc()
        log.exception("inference failed")
        raise HTTPException(500, "Inference failed") from exc
    thr = s.meta["threshold"]
    out = []
    for row, p in zip(rows, probas, strict=True):
        rid, p = uuid.uuid4().hex[:12], float(p)
        m.PROBABILITY.observe(p)
        m.PREDICTIONS.labels("churn" if p >= thr else "stay").inc()
        log_prediction(row, p, rid)
        out.append(Prediction(churn_probability=round(p, 4), churn=p >= thr,
                              risk_level=risk_level(p), model_version=s.meta["version"],
                              request_id=rid))
    return out


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    store.load()
    yield


app = FastAPI(
    title="Telco Churn Prediction API", version=__version__, lifespan=lifespan,
    description="Predict customer churn probability, explain predictions (SHAP) and "
                "expose Prometheus metrics. Part of the DDM501 end-to-end MLOps project.",
    responses={503: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)


@app.middleware("http")
async def observe(request: Request, call_next: Any) -> Response:
    start = time.perf_counter()
    status = 500
    try:
        response: Response = await call_next(request)
        status = response.status_code
        return response
    finally:
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")
        if path != "/metrics":
            m.HTTP_REQUESTS.labels(request.method, path, str(status)).inc()
            m.HTTP_LATENCY.labels(request.method, path).observe(time.perf_counter() - start)


@app.get("/health", tags=["ops"], summary="Liveness probe")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready", tags=["ops"], summary="Readiness probe (model loaded?)")
def ready() -> dict[str, Any]:
    if not store.ready:
        raise HTTPException(503, "Model not loaded")
    return {"status": "ready", "model_version": store.meta["version"]}


@app.get("/metrics", tags=["ops"], include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/predict", response_model=Prediction, tags=["prediction"])
def predict(customer: Customer) -> Prediction:
    """Churn probability for a single customer."""
    return _predict([customer])[0]


@app.post("/v1/predict/batch", response_model=BatchResponse, tags=["prediction"])
def predict_batch(req: BatchRequest) -> BatchResponse:
    """Churn probabilities for up to 500 customers."""
    return BatchResponse(predictions=_predict(req.customers))


@app.post("/v1/explain", response_model=Explanation, tags=["explainability"])
def explain(customer: Customer, top_k: int = 5) -> Explanation:
    """Prediction + the top-k SHAP feature contributions."""
    s = _require_model()
    if s.explainer is None:
        raise HTTPException(503, "Explainer unavailable (no reference data)")
    pred = _predict([customer])[0]
    df = pd.DataFrame([customer.model_dump()])[s.meta["feature_columns"]]
    top = s.explainer.top_features(df, max(1, min(top_k, 15)))
    return Explanation(prediction=pred, top_features=[FeatureContribution(**t) for t in top])


@app.get("/v1/model", response_model=ModelInfo, tags=["model"])
def model_info() -> ModelInfo:
    s = _require_model()
    return ModelInfo(version=s.meta["version"], model_type=s.meta["model_type"],
                     trained_at=s.meta["trained_at"], metrics=s.meta["metrics"],
                     git_sha=s.meta.get("git_sha", "unknown"), threshold=s.meta["threshold"])


def admin_auth(x_admin_token: str | None = Header(default=None)) -> None:
    expected = os.getenv("ADMIN_TOKEN")
    if not expected:
        raise HTTPException(503, "Admin endpoint disabled (ADMIN_TOKEN not set)")
    if x_admin_token != expected:
        raise HTTPException(403, "Invalid admin token")


@app.post("/v1/admin/reload", tags=["admin"], dependencies=[Depends(admin_auth)])
def reload_model() -> dict[str, Any]:
    """Hot-reload the champion model (called by Airflow after a promotion)."""
    if not store.load():
        raise HTTPException(500, "Reload failed")
    return {"status": "reloaded", "model_version": store.meta["version"]}
