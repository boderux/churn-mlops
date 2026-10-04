import pandas as pd
import pytest

from churn import data


def test_validate_ok(raw_df):
    assert data.validate(raw_df) == []


def test_validate_missing_column(raw_df):
    assert "missing columns" in data.validate(raw_df.drop(columns=["tenure"]))[0]


def test_validate_out_of_range(raw_df):
    bad = raw_df.copy()
    bad.loc[0, "tenure"] = 999
    assert any("out-of-range" in e for e in data.validate(bad))


def test_validate_unexpected_category(raw_df):
    bad = raw_df.copy()
    bad.loc[0, "Contract"] = "Weekly"
    assert any("unexpected values in Contract" in e for e in data.validate(bad))


def test_validate_duplicates_and_nulls(raw_df):
    bad = pd.concat([raw_df, raw_df.head(5)])
    assert any("duplicate" in e for e in data.validate(bad))
    bad = raw_df.copy()
    bad.loc[:200, "Partner"] = None
    assert any("null rate" in e for e in data.validate(bad))


def test_clean_blank_total_charges():
    df = data.generate_synthetic(10)
    df["TotalCharges"] = df["TotalCharges"].astype(object)
    df.loc[0, "TotalCharges"] = " "
    out = data.clean(df)
    assert out.loc[0, "TotalCharges"] == 0.0
    assert out["SeniorCitizen"].isin(["0", "1"]).all()


def test_preprocess_rejects_bad_data(params, raw_df, tmp_path):
    p = {**params, "data": {**params["data"], "raw_path": str(tmp_path / "bad.csv")}}
    raw_df.drop(columns=["tenure"]).to_csv(p["data"]["raw_path"], index=False)
    with pytest.raises(ValueError, match="missing columns"):
        data.preprocess(p)


def test_preprocess_rejects_invalid_values(params, raw_df, tmp_path):
    p = {**params, "data": {**params["data"], "raw_path": str(tmp_path / "bad2.csv")}}
    bad = raw_df.copy()
    bad["Contract"] = "Weekly"
    bad.to_csv(p["data"]["raw_path"], index=False)
    with pytest.raises(ValueError, match="validation failed"):
        data.preprocess(p)


def test_ingest_uses_existing_file(params):
    assert data.ingest(params).exists()


def test_ingest_synthetic_fallback(params, tmp_path):
    p = {
        **params,
        "data": {**params["data"], "raw_path": str(tmp_path / "x.csv"), "url": "http://127.0.0.1:9/none"},
    }
    out = data.ingest(p)
    assert len(pd.read_csv(out)) > 1000
    p["data"]["allow_synthetic_fallback"] = False
    out.unlink()
    with pytest.raises(Exception):  # noqa: B017
        data.ingest(p)
