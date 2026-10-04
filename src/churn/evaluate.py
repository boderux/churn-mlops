"""Model validation gate: decide whether the candidate may replace the champion."""

from __future__ import annotations

import json
import logging
from typing import Any

from churn.config import ARTIFACT_DIR, CHAMPION_DIR

log = logging.getLogger(__name__)


def gate(params: dict[str, Any]) -> dict[str, Any]:
    """Absolute thresholds + non-regression vs. current champion."""
    g = params["gate"]
    cand = json.loads((ARTIFACT_DIR / "candidate_metrics.json").read_text())
    m = cand["metrics"]
    reasons: list[str] = []
    if m["roc_auc"] < g["min_roc_auc"]:
        reasons.append(f"roc_auc {m['roc_auc']:.3f} < {g['min_roc_auc']}")
    if m["recall"] < g["min_recall"]:
        reasons.append(f"recall {m['recall']:.3f} < {g['min_recall']}")
    champ_auc = None
    champ_file = CHAMPION_DIR / "metadata.json"
    if champ_file.exists():
        champ_auc = json.loads(champ_file.read_text())["metrics"]["roc_auc"]
        if m["roc_auc"] < champ_auc - g["max_regression"]:
            reasons.append(f"regression vs champion: {m['roc_auc']:.3f} < {champ_auc:.3f}")
    decision = {
        "promote": not reasons,
        "reasons": reasons,
        "candidate_auc": m["roc_auc"],
        "champion_auc": champ_auc,
        "model_type": cand["model_type"],
        "version": cand["version"],
    }
    (ARTIFACT_DIR / "gate_decision.json").write_text(json.dumps(decision, indent=2))
    log.info("Gate decision: %s", decision)
    return decision
