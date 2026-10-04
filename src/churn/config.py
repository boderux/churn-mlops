"""Configuration loading and shared constants."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", ROOT / "artifacts"))
CHAMPION_DIR = ARTIFACT_DIR / "champion"

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
RAW_FEATURES = NUMERIC_COLS + CATEGORICAL_COLS
SERVICE_COLS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def load_params(path: str | Path | None = None) -> dict[str, Any]:
    """Load pipeline parameters from YAML (path overridable via PARAMS_PATH)."""
    env_path = os.getenv("PARAMS_PATH")
    p = Path(path or env_path or ROOT / "configs" / "params.yaml")
    with open(p, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(path: str | Path) -> Path:
    """Resolve a repo-relative path against ROOT."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
