import json
from unittest.mock import MagicMock, patch

import requests

from churn import evaluate, model_io
from churn.config import ARTIFACT_DIR, CHAMPION_DIR
from churn.notify import send_telegram


def test_telegram_noop_without_secrets():
    assert send_telegram("hello") is False


def test_telegram_sends(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "1")
    ok = MagicMock(raise_for_status=lambda: None)
    with patch("churn.notify.requests.post", return_value=ok) as post:
        assert send_telegram("drift!", "critical") is True
    assert "🚨" in post.call_args.kwargs["json"]["text"]


def test_telegram_failure_never_raises(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "1")
    with patch("churn.notify.requests.post", side_effect=requests.ConnectionError("x")):
        assert send_telegram("x") is False


def _write_candidate(auc, recall=0.8):
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "candidate_metrics.json").write_text(json.dumps({
        "metrics": {"roc_auc": auc, "recall": recall}, "model_type": "t", "version": "v"}))


def test_gate_rejects_low_auc(params, trained):
    _write_candidate(0.40)
    d = evaluate.gate(params)
    assert not d["promote"] and any("roc_auc" in r for r in d["reasons"])


def test_gate_rejects_regression(params, trained):
    champ = json.loads((CHAMPION_DIR / "metadata.json").read_text())["metrics"]["roc_auc"]
    _write_candidate(champ - 0.05)
    d = evaluate.gate(params)
    assert not d["promote"] and any("regression" in r for r in d["reasons"])


def test_gate_accepts_better(params, trained):
    _write_candidate(0.99)
    assert evaluate.gate(params)["promote"]


def test_bundle_roundtrip(trained):
    pipe, meta = model_io.load_bundle(CHAMPION_DIR)
    assert meta["version"] == trained["version"]
    assert model_io.load_reference(CHAMPION_DIR) is not None
    assert hasattr(pipe, "predict_proba")
