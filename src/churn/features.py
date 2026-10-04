"""Feature engineering + preprocessing pipeline."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from churn.config import CATEGORICAL_COLS, NUMERIC_COLS, SERVICE_COLS

ENGINEERED_NUM = ["avg_monthly_spend", "n_services", "charge_ratio"]
ENGINEERED_CAT = ["tenure_group", "is_month_to_month", "has_internet"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Domain features. Module-level function => picklable inside Pipeline."""
    df = df.copy()
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int).astype(str)
    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["charge_ratio"] = df["MonthlyCharges"] / (df["avg_monthly_spend"] + 1e-6)
    df["n_services"] = sum((df[c] == "Yes").astype(int) for c in SERVICE_COLS)
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, 1000], labels=["0-12m", "13-24m", "25-48m", "49m+"]
    ).astype(str)
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int).astype(str)
    df["has_internet"] = (df["InternetService"] != "No").astype(int).astype(str)
    return df


def build_preprocessor(drop: list[str] | None = None) -> ColumnTransformer:
    drop = drop or []
    num = [c for c in NUMERIC_COLS + ENGINEERED_NUM if c not in drop]
    cat = [c for c in CATEGORICAL_COLS + ENGINEERED_CAT if c not in drop]
    return ColumnTransformer(
        [
            ("num", StandardScaler(), num),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ]
    )


def feature_engineering_step() -> FunctionTransformer:
    return FunctionTransformer(add_features, validate=False)
