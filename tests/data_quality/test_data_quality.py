"""Data-quality tests on the raw + processed datasets."""

import pandas as pd

from churn import data
from churn.config import CATEGORICAL_COLS, NUMERIC_COLS


def test_schema_complete(raw_df):
    assert set(NUMERIC_COLS + CATEGORICAL_COLS + ["Churn", "customerID"]) <= set(raw_df.columns)


def test_no_nulls_in_features(raw_df):
    assert raw_df[NUMERIC_COLS + CATEGORICAL_COLS].isna().sum().sum() == 0


def test_numeric_ranges(raw_df):
    assert raw_df["tenure"].between(0, 120).all()
    assert raw_df["MonthlyCharges"].between(0, 200).all()
    assert (raw_df["TotalCharges"] >= 0).all()


def test_target_is_binary_and_not_degenerate(raw_df):
    assert set(raw_df["Churn"].unique()) <= {"Yes", "No"}
    assert 0.05 < (raw_df["Churn"] == "Yes").mean() < 0.6


def test_logical_consistency(raw_df):
    no_net = raw_df[raw_df["InternetService"] == "No"]
    assert (no_net["TechSupport"] == "No internet service").all()
    no_phone = raw_df[raw_df["PhoneService"] == "No"]
    assert (no_phone["MultipleLines"] == "No phone service").all()


def test_split_has_no_leakage_and_is_stratified(trained, params):
    d = params["data"]["processed_dir"]
    tr, te = pd.read_csv(f"{d}/train.csv"), pd.read_csv(f"{d}/test.csv")
    assert set(tr["customerID"]).isdisjoint(te["customerID"])
    assert abs((tr["Churn"] == "Yes").mean() - (te["Churn"] == "Yes").mean()) < 0.02
    assert abs(len(te) / (len(tr) + len(te)) - params["data"]["test_size"]) < 0.01


def test_real_dataset_passes_validation_if_downloaded():
    from churn.config import ROOT
    f = ROOT / "data" / "raw" / "telco_churn.csv"
    if f.exists():
        assert data.validate(data.clean(pd.read_csv(f))) == []
