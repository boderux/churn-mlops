"""Data ingestion, validation and splitting."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from churn.config import CATEGORICAL_COLS, NUMERIC_COLS, resolve

log = logging.getLogger(__name__)

ALLOWED_VALUES: dict[str, set[Any]] = {
    "gender": {"Male", "Female"},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "InternetService": {"DSL", "Fiber optic", "No"},
}
RANGES = {"tenure": (0, 120), "MonthlyCharges": (0, 200), "TotalCharges": (0, 15000)}


def generate_synthetic(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Telco-like synthetic data (offline fallback + test fixtures)."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 73, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    df = pd.DataFrame(
        {
            "customerID": [f"C{i:05d}" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], n),
            "SeniorCitizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
            "Partner": rng.choice(["Yes", "No"], n),
            "Dependents": rng.choice(["Yes", "No"], n, p=[0.3, 0.7]),
            "tenure": tenure,
            "PhoneService": rng.choice(["Yes", "No"], n, p=[0.9, 0.1]),
            "InternetService": internet,
            "Contract": contract,
            "PaperlessBilling": rng.choice(["Yes", "No"], n, p=[0.6, 0.4]),
            "PaymentMethod": rng.choice(
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                n,
            ),
        }
    )
    for col in [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]:
        df[col] = np.where(df["InternetService"] == "No", "No internet service", rng.choice(["Yes", "No"], n))
    df["MultipleLines"] = np.where(
        df["PhoneService"] == "No", "No phone service", rng.choice(["Yes", "No"], n)
    )
    base = 20 + 35 * (internet != "No") + 25 * (internet == "Fiber optic")
    df["MonthlyCharges"] = np.round(base + rng.normal(0, 6, n).clip(-10, 15), 2)
    df["TotalCharges"] = np.round(df["MonthlyCharges"] * tenure, 2)
    logit = (
        -1.0
        + 1.4 * (contract == "Month-to-month")
        - 1.1 * (contract == "Two year")
        + 0.7 * (internet == "Fiber optic")
        - 0.035 * tenure
        + 0.4 * (df["PaymentMethod"] == "Electronic check")
        + 0.25 * df["SeniorCitizen"]
        - 0.5 * (df["TechSupport"] == "Yes")
        + rng.normal(0, 0.6, n)
    )
    df["Churn"] = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")
    return df


def ingest(params: dict[str, Any]) -> Path:
    """Download the raw dataset (or fallback to synthetic) and store it."""
    cfg = params["data"]
    out = resolve(cfg["raw_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        log.info("Raw data already present: %s", out)
        return out
    try:
        resp = requests.get(cfg["url"], timeout=30)
        resp.raise_for_status()
        out.write_bytes(resp.content)
        log.info("Downloaded dataset (%d bytes)", len(resp.content))
    except Exception as exc:  # noqa: BLE001
        if not cfg.get("allow_synthetic_fallback", False):
            raise
        log.warning("Download failed (%s) -> synthetic fallback", exc)
        generate_synthetic().to_csv(out, index=False)
    return out


def load_raw(params: dict[str, Any]) -> pd.DataFrame:
    """Load and clean the raw CSV (TotalCharges blanks -> 0 for new customers)."""
    df = pd.read_csv(resolve(params["data"]["raw_path"]))
    return clean(df)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    if "SeniorCitizen" in df:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(int).astype(str)
    return df


def validate(df: pd.DataFrame, target: str | None = "Churn") -> list[str]:
    """Return a list of data-quality violations (empty list == OK)."""
    errors: list[str] = []
    needed = NUMERIC_COLS + CATEGORICAL_COLS + ([target] if target else [])
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return [f"missing columns: {missing}"]
    if len(df) < 500 and target:
        errors.append(f"too few rows: {len(df)}")
    null_rate = df[needed].isna().mean()
    for col, rate in null_rate[null_rate > 0.02].items():
        errors.append(f"null rate {rate:.1%} in {col}")
    for col, (lo, hi) in RANGES.items():
        bad = int(((df[col] < lo) | (df[col] > hi)).sum())
        if bad:
            errors.append(f"{bad} out-of-range values in {col} [{lo},{hi}]")
    for col, allowed in ALLOWED_VALUES.items():
        bad_vals = set(df[col].dropna().unique()) - allowed
        if bad_vals:
            errors.append(f"unexpected values in {col}: {sorted(bad_vals)}")
    if "customerID" in df and df["customerID"].duplicated().any():
        errors.append("duplicate customerID")
    if target:
        rate = (df[target] == "Yes").mean()
        if not 0.05 <= rate <= 0.60:
            errors.append(f"suspicious churn rate: {rate:.1%}")
    return errors


def split(df: pd.DataFrame, params: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = params["data"]
    train, test = train_test_split(
        df, test_size=cfg["test_size"], random_state=cfg["random_state"], stratify=df[cfg["target"]]
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def preprocess(params: dict[str, Any]) -> dict[str, Path]:
    """validate -> split -> persist train/test (reference data for drift)."""
    df = load_raw(params)
    errors = validate(df, params["data"]["target"])
    if errors:
        raise ValueError("Data validation failed: " + "; ".join(errors))
    train, test = split(df, params)
    out = resolve(params["data"]["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    paths = {"train": out / "train.csv", "test": out / "test.csv"}
    train.to_csv(paths["train"], index=False)
    test.to_csv(paths["test"], index=False)
    log.info("train=%d test=%d", len(train), len(test))
    try:
        from churn.storage import sync_data_to_minio

        sync_data_to_minio(params)
    except Exception as exc:  # noqa: BLE001
        log.debug("Data sync to MinIO skipped: %s", exc)
    return paths
