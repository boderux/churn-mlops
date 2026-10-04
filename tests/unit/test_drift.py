import json

import pytest
import yaml

from churn import drift
from churn.cli import main
from churn.config import ARTIFACT_DIR
from churn.model_io import load_reference

pytestmark = pytest.mark.slow


def _shift(ref):
    cur = ref.copy()
    cur["tenure"], cur["Contract"], cur["churn_probability"] = 1, "Month-to-month", 0.95
    cur["MonthlyCharges"] = cur["MonthlyCharges"] + 80
    cur["InternetService"] = "Fiber optic"
    cur["PaymentMethod"] = "Electronic check"
    return cur


@pytest.fixture()
def ref(trained):
    r = load_reference().copy()
    r["churn_probability"] = 0.3
    return r


def test_no_drift_on_same_distribution(ref):
    cur = ref.sample(len(ref), replace=True, random_state=1)
    res, _ = drift.compute_drift(ref, cur, 0.3)
    assert res["dataset_drift"] is False


def test_drift_detected_when_shifted(ref):
    res, _ = drift.compute_drift(ref, _shift(ref), 0.3)
    assert res["dataset_drift"] and "tenure" in res["drifted_columns"]


def test_run_skips_without_enough_data(params, ref):
    assert drift.run(params, current=ref.head(5))["skipped"] is True


def test_run_writes_report_and_alerts(params, ref):
    res = drift.run(params, current=_shift(ref))
    assert (ARTIFACT_DIR / "reports" / "drift_report.html").exists()
    assert res["dataset_drift"] is True
    assert drift.alert_if_drift(res) is False  # no token configured -> only logged
    assert drift.alert_if_drift({"dataset_drift": False}) is False


def test_load_current_from_log(params, tmp_path, ref):
    f = tmp_path / "log.jsonl"
    rec = {
        "features": {**ref.iloc[0].drop("churn_probability").to_dict(), "SeniorCitizen": 0},
        "probability": 0.4,
    }
    f.write_text(json.dumps(rec) + "\nnot-json\n")
    p = {**params, "monitoring": {**params["monitoring"], "prediction_log": str(f)}}
    df = drift.load_current(p)
    assert len(df) == 1 and df["SeniorCitizen"].iloc[0] == "0"
    p["monitoring"]["prediction_log"] = str(tmp_path / "missing.jsonl")
    assert drift.load_current(p).empty


def test_push_metrics_called(monkeypatch):
    called = {}
    monkeypatch.setattr(drift, "push_to_gateway", lambda url, **kw: called.update(url=url, **kw))
    drift.push_metrics(
        {"share_drifted": 0.5, "dataset_drift": True, "n_drifted": 3, "n_current": 200}, "http://pg:9091"
    )
    assert called["job"] == "churn_drift_monitor"


def test_cli_commands(trained, params, tmp_path):
    pf = tmp_path / "p.yaml"
    pf.write_text(yaml.safe_dump(params))
    for cmd in ["validate", "gate", "promote"]:
        assert main([cmd, "--params", str(pf)]) == 0
