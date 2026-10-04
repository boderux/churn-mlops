"""Fairness analysis (Fairlearn) + optional post-processing mitigation."""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.metrics import accuracy_score

from churn.config import ARTIFACT_DIR, CHAMPION_DIR, resolve
from churn.model_io import load_bundle

log = logging.getLogger(__name__)
FOUR_FIFTHS = 0.8


def analyse_group(y_true: np.ndarray, y_pred: np.ndarray, groups: pd.Series) -> dict[str, Any]:
    mf = MetricFrame(
        metrics={
            "selection_rate": selection_rate,
            "tpr": true_positive_rate,
            "fpr": false_positive_rate,
            "accuracy": accuracy_score,
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=groups,
    )
    sel = mf.by_group["selection_rate"]
    ratio = float(sel.min() / sel.max()) if sel.max() > 0 else 1.0
    return {
        "by_group": {
            str(k): {m: round(float(v), 4) for m, v in row.items()} for k, row in mf.by_group.iterrows()
        },
        "demographic_parity_diff": round(
            float(demographic_parity_difference(y_true, y_pred, sensitive_features=groups)), 4
        ),
        "equalized_odds_diff": round(
            float(equalized_odds_difference(y_true, y_pred, sensitive_features=groups)), 4
        ),
        "disparate_impact_ratio": round(ratio, 4),
        "passes_four_fifths_rule": ratio >= FOUR_FIFTHS,
    }


def run(params: dict[str, Any], model_dir=None) -> dict[str, Any]:
    """Audit the model on the test set for every sensitive attribute and try mitigation."""
    mdir = model_dir or (
        CHAMPION_DIR if (CHAMPION_DIR / "model.joblib").exists() else ARTIFACT_DIR / "candidate"
    )
    pipe, meta = load_bundle(mdir)
    proc = resolve(params["data"]["processed_dir"])
    dtype = {"SeniorCitizen": str}
    train = pd.read_csv(proc / "train.csv", dtype=dtype)
    test = pd.read_csv(proc / "test.csv", dtype=dtype)
    feats, target = meta["feature_columns"], params["data"]["target"]
    y = (test[target] == "Yes").astype(int).to_numpy()
    proba = pipe.predict_proba(test[feats])[:, 1]
    pred = (proba >= meta["threshold"]).astype(int)

    report: dict[str, Any] = {"model_version": meta["version"], "attributes": {}}
    for attr in params["features"]["sensitive_attributes"]:
        res = {"baseline": analyse_group(y, pred, test[attr].astype(str))}
        try:  # mitigation: group-specific thresholds (equalized odds)
            to = ThresholdOptimizer(
                estimator=pipe,
                constraints="equalized_odds",
                objective="balanced_accuracy_score",
                prefit=True,
                predict_method="predict_proba",
            )
            to.fit(
                train[feats], (train[target] == "Yes").astype(int), sensitive_features=train[attr].astype(str)
            )
            mitigated = to.predict(test[feats], sensitive_features=test[attr].astype(str), random_state=42)
            res["mitigated_equalized_odds"] = analyse_group(y, np.asarray(mitigated), test[attr].astype(str))
        except Exception as exc:  # noqa: BLE001
            log.warning("Mitigation failed for %s: %s", attr, exc)
        report["attributes"][attr] = res
    out = ARTIFACT_DIR / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / "fairness_report.json").write_text(json.dumps(report, indent=2))
    log.info("Fairness report saved to %s", out / "fairness_report.json")
    try:
        from churn.storage import sync_reports_to_minio

        sync_reports_to_minio()
    except Exception as exc:  # noqa: BLE001
        log.debug("Fairness report sync to MinIO skipped: %s", exc)
    return report
