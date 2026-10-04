"""Data & prediction drift monitoring with Evidently -> Prometheus Pushgateway -> Telegram."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from churn.config import ARTIFACT_DIR, CATEGORICAL_COLS, CHAMPION_DIR, NUMERIC_COLS, resolve
from churn.model_io import load_bundle, load_reference
from churn.notify import send_telegram

log = logging.getLogger(__name__)


def load_current(params: dict[str, Any]) -> pd.DataFrame:
    """Flatten the API prediction log (JSON lines) into a dataframe."""
    path = resolve(params["monitoring"]["prediction_log"])
    if not path.exists():
        return pd.DataFrame()
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
                rows.append({**rec["features"], "churn_probability": rec["probability"]})
            except (json.JSONDecodeError, KeyError):
                continue
    df = pd.DataFrame(rows)
    if "SeniorCitizen" in df:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(int).astype(str)
    return df


def compute_drift(
    reference: pd.DataFrame, current: pd.DataFrame, threshold: float
) -> tuple[dict[str, Any], Report]:
    cols = [c for c in reference.columns if c in current.columns]
    mapping = ColumnMapping(
        numerical_features=[c for c in cols if c in NUMERIC_COLS or c == "churn_probability"],
        categorical_features=[c for c in cols if c in CATEGORICAL_COLS],
    )
    report = Report(metrics=[DataDriftPreset(drift_share=threshold)])
    report.run(reference_data=reference[cols], current_data=current[cols], column_mapping=mapping)
    res = report.as_dict()["metrics"]
    summary = res[0]["result"]
    by_col = res[1]["result"]["drift_by_columns"]
    drifted = sorted(c for c, v in by_col.items() if v["drift_detected"])
    return {
        "dataset_drift": bool(summary["dataset_drift"]),
        "share_drifted": float(summary["share_of_drifted_columns"]),
        "n_drifted": int(summary["number_of_drifted_columns"]),
        "n_columns": int(summary["number_of_columns"]),
        "drifted_columns": drifted,
        "n_reference": len(reference),
        "n_current": len(current),
    }, report


def push_metrics(result: dict[str, Any], url: str) -> None:
    reg = CollectorRegistry()
    for name, doc, val in [
        ("churn_data_drift_share", "Share of drifted columns", result["share_drifted"]),
        ("churn_data_drift_detected", "1 if dataset drift detected", float(result["dataset_drift"])),
        ("churn_drifted_columns", "Number of drifted columns", result["n_drifted"]),
        ("churn_drift_current_rows", "Rows in current window", result["n_current"]),
        ("churn_drift_last_run_timestamp", "Unix time of last drift check", time.time()),
    ]:
        Gauge(name, doc, registry=reg).set(val)
    push_to_gateway(url, job="churn_drift_monitor", registry=reg)


def run(params: dict[str, Any], current: pd.DataFrame | None = None) -> dict[str, Any]:
    mcfg = params["monitoring"]
    current = load_current(params) if current is None else current
    if len(current) < mcfg["min_current_rows"]:
        msg = f"Not enough production data for drift check ({len(current)}<{mcfg['min_current_rows']})"
        log.warning(msg)
        return {"skipped": True, "reason": msg, "dataset_drift": False, "n_current": len(current)}
    reference = load_reference(CHAMPION_DIR)
    if reference is None:
        raise FileNotFoundError("No reference data: train and promote a model first")
    pipe, meta = load_bundle(CHAMPION_DIR)
    reference = reference.copy()
    reference["churn_probability"] = pipe.predict_proba(reference[meta["feature_columns"]])[:, 1]
    result, report = compute_drift(reference, current, mcfg["drift_share_threshold"])
    out = ARTIFACT_DIR / "reports"
    out.mkdir(parents=True, exist_ok=True)
    report.save_html(str(out / "drift_report.html"))
    (out / "drift_summary.json").write_text(json.dumps(result, indent=2))
    try:
        from churn.storage import sync_reports_to_minio

        sync_reports_to_minio()
    except Exception as exc:  # noqa: BLE001
        log.debug("Report sync to MinIO skipped: %s", exc)
    url = os.getenv("PUSHGATEWAY_URL") or mcfg.get("pushgateway_url")
    if url:
        try:
            push_metrics(result, url)
        except Exception as exc:  # noqa: BLE001
            log.warning("Pushgateway unavailable: %s", exc)
    log.info("Drift result: %s", result)
    return result


def alert_if_drift(result: dict[str, Any]) -> bool:
    """Telegram alert when dataset drift is detected."""
    if not result.get("dataset_drift"):
        return False
    cols = ", ".join(result["drifted_columns"][:8]) or "-"
    return send_telegram(
        f"<b>Data drift detected</b>\nDrifted columns: {result['n_drifted']}/{result['n_columns']} "
        f"({result['share_drifted']:.0%})\nTop: <code>{cols}</code>\n"
        f"Window: {result['n_current']} requests\n➡️ Retraining pipeline triggered.",
        level="warning",
    )
