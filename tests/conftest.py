"""Shared fixtures. Artifacts are redirected to a temp dir BEFORE importing the package."""

import os
import sys
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="churn-test-"))
os.environ["ARTIFACT_DIR"] = str(_TMP / "artifacts")
os.environ["PREDICTION_LOG"] = str(_TMP / "artifacts" / "prediction_log.jsonl")
os.environ["ADMIN_TOKEN"] = "test-token"
os.environ.pop("TELEGRAM_BOT_TOKEN", None)
os.environ.pop("TELEGRAM_CHAT_ID", None)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from churn import data, evaluate, model_io, train  # noqa: E402
from churn.config import CHAMPION_DIR, load_params  # noqa: E402

SAMPLE_CUSTOMER = {
    "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No", "tenure": 5,
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
    "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "Yes",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 89.1, "TotalCharges": 445.5,
}


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return data.clean(data.generate_synthetic(2500, seed=1))


@pytest.fixture(scope="session")
def params(raw_df) -> dict:
    p = load_params()
    p["data"]["raw_path"] = str(_TMP / "raw.csv")
    p["data"]["processed_dir"] = str(_TMP / "processed")
    p["training"]["models"] = ["logreg", "xgboost"]
    p["gate"].update(min_roc_auc=0.6, min_recall=0.3)
    raw_df.to_csv(p["data"]["raw_path"], index=False)
    return p


@pytest.fixture(scope="session")
def trained(params) -> dict:
    """Run preprocess -> train -> gate -> promote once for the whole session."""
    data.preprocess(params)
    meta = train.train(params, quick=True)
    decision = evaluate.gate(params)
    assert decision["promote"], decision
    model_io.promote(train.CANDIDATE_DIR, CHAMPION_DIR)
    return meta


@pytest.fixture(scope="session")
def client(trained):
    from fastapi.testclient import TestClient

    from api.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def customer() -> dict:
    return dict(SAMPLE_CUSTOMER)
