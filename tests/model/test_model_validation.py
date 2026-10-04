"""Model validation tests: quality floor, behaviour, robustness, latency, RAI artefacts."""

import time

import numpy as np
import pandas as pd
import pytest

from churn import explain, fairness
from churn.config import CHAMPION_DIR
from churn.model_io import load_bundle

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def pipe_meta(trained):
    return load_bundle(CHAMPION_DIR)


def _row(customer, **kw):
    return pd.DataFrame([{**customer, **kw}])


def test_quality_floor(pipe_meta):
    _, meta = pipe_meta
    assert meta["metrics"]["roc_auc"] >= 0.70
    assert meta["metrics"]["recall"] >= 0.50


def test_probabilities_valid(pipe_meta, raw_df):
    pipe, meta = pipe_meta
    p = pipe.predict_proba(raw_df[meta["feature_columns"]].head(200))[:, 1]
    assert np.all((p >= 0) & (p <= 1)) and not np.isnan(p).any()


def test_directional_contract(pipe_meta, customer):
    pipe, meta = pipe_meta
    f = meta["feature_columns"]
    m2m = pipe.predict_proba(_row(customer, Contract="Month-to-month")[f])[0, 1]
    two = pipe.predict_proba(_row(customer, Contract="Two year")[f])[0, 1]
    assert m2m > two


def test_directional_tenure(pipe_meta, customer):
    pipe, meta = pipe_meta
    f = meta["feature_columns"]
    new = pipe.predict_proba(_row(customer, tenure=1, TotalCharges=89.1)[f])[0, 1]
    old = pipe.predict_proba(_row(customer, tenure=70, TotalCharges=89.1 * 70)[f])[0, 1]
    assert new > old


def test_sensitive_attribute_not_used(pipe_meta):
    _, meta = pipe_meta
    assert "gender" not in meta["feature_columns"]


def test_handles_unseen_category(pipe_meta, customer):
    pipe, meta = pipe_meta
    row = _row(customer, PaymentMethod="Crypto")[meta["feature_columns"]]
    assert 0 <= pipe.predict_proba(row)[0, 1] <= 1


def test_single_prediction_latency(pipe_meta, customer):
    pipe, meta = pipe_meta
    row = _row(customer)[meta["feature_columns"]]
    pipe.predict_proba(row)
    t = time.perf_counter()
    for _ in range(20):
        pipe.predict_proba(row)
    assert (time.perf_counter() - t) / 20 < 0.1


def test_fairness_report(params, trained):
    rep = fairness.run(params)
    for attr in ["gender", "SeniorCitizen"]:
        base = rep["attributes"][attr]["baseline"]
        assert 0 <= base["disparate_impact_ratio"] <= 1
        assert {"demographic_parity_diff", "equalized_odds_diff", "by_group"} <= set(base)


def test_explainability_artifacts(params, trained):
    res = explain.run(params)
    assert len(res["global_top_features"]) == 10
    assert res["local_shap"] and res["local_lime"]
