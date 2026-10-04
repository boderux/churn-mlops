import json
import os

import pytest

from api.main import store

pytestmark = pytest.mark.slow


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    r = client.get("/ready")
    assert r.status_code == 200 and r.json()["status"] == "ready"


def test_predict_single(client, customer):
    r = client.post("/v1/predict", json=customer)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["risk_level"] in {"low", "medium", "high"}
    assert body["churn"] == (body["churn_probability"] >= 0.5)


def test_predict_batch(client, customer):
    r = client.post("/v1/predict/batch", json={"customers": [customer, {**customer, "tenure": 60}]})
    assert r.status_code == 200 and len(r.json()["predictions"]) == 2


@pytest.mark.parametrize("patch", [
    {"tenure": -1}, {"Contract": "Weekly"}, {"MonthlyCharges": 1e6}, {"gender": "Male"},
])
def test_predict_validation_errors(client, customer, patch):
    assert client.post("/v1/predict", json={**customer, **patch}).status_code == 422


def test_missing_field(client, customer):
    customer.pop("tenure")
    assert client.post("/v1/predict", json=customer).status_code == 422


def test_batch_limits(client):
    assert client.post("/v1/predict/batch", json={"customers": []}).status_code == 422


def test_explain(client, customer):
    r = client.post("/v1/explain?top_k=3", json=customer)
    assert r.status_code == 200
    feats = r.json()["top_features"]
    assert len(feats) == 3 and {"feature", "shap", "direction"} <= set(feats[0])


def test_model_info(client):
    r = client.get("/v1/model")
    assert r.status_code == 200 and r.json()["metrics"]["roc_auc"] > 0.5


def test_metrics_endpoint(client, customer):
    client.post("/v1/predict", json=customer)
    text = client.get("/metrics").text
    for name in ["churn_http_requests_total", "churn_predictions_total", "churn_model_roc_auc",
                 "churn_http_request_duration_seconds_bucket", "churn_model_loaded 1.0"]:
        assert name in text


def test_prediction_logged_for_drift(client, customer):
    client.post("/v1/predict", json=customer)
    lines = open(os.environ["PREDICTION_LOG"]).read().splitlines()
    rec = json.loads(lines[-1])
    assert "gender" not in rec["features"] and 0 <= rec["probability"] <= 1


def test_admin_reload_auth(client):
    assert client.post("/v1/admin/reload").status_code == 403
    assert client.post("/v1/admin/reload", headers={"X-Admin-Token": "bad"}).status_code == 403
    ok = client.post("/v1/admin/reload", headers={"X-Admin-Token": "test-token"})
    assert ok.status_code == 200 and ok.json()["status"] == "reloaded"


def test_admin_disabled_without_token(client, monkeypatch):
    monkeypatch.delenv("ADMIN_TOKEN")
    assert client.post("/v1/admin/reload").status_code == 503


def test_503_when_model_missing(client, customer, tmp_path):
    saved = (store.pipe, store.meta, store.explainer)
    store.pipe = None
    try:
        assert client.get("/ready").status_code == 503
        assert client.post("/v1/predict", json=customer).status_code == 503
        assert client.get("/v1/model").status_code == 503
        assert store.load(tmp_path) is False
    finally:
        store.pipe, store.meta, store.explainer = saved


def test_inference_failure_returns_500(client, customer, monkeypatch):
    class Boom:
        def predict_proba(self, _):
            raise RuntimeError("boom")
    monkeypatch.setattr(store, "pipe", Boom())
    assert client.post("/v1/predict", json=customer).status_code == 500
