"""Pydantic request/response models (drive the OpenAPI docs)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

YesNo = Literal["Yes", "No"]
YesNoNoInternet = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    """One customer. NOTE: `gender` is intentionally NOT collected (data minimisation)."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{
        "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No", "tenure": 5,
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "Yes",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 89.1, "TotalCharges": 445.5}]})

    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="Months with the company")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: YesNoNoInternet
    OnlineBackup: YesNoNoInternet
    DeviceProtection: YesNoNoInternet
    TechSupport: YesNoNoInternet
    StreamingTV: YesNoNoInternet
    StreamingMovies: YesNoNoInternet
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal["Electronic check", "Mailed check",
                           "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float = Field(ge=0, le=200)
    TotalCharges: float = Field(ge=0, le=15000)


class Prediction(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    churn_probability: float = Field(ge=0, le=1)
    churn: bool
    risk_level: Literal["low", "medium", "high"]
    model_version: str
    request_id: str


class BatchRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=500)


class BatchResponse(BaseModel):
    predictions: list[Prediction]


class FeatureContribution(BaseModel):
    feature: str
    value: float
    shap: float
    direction: str


class Explanation(BaseModel):
    prediction: Prediction
    top_features: list[FeatureContribution]


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    version: str
    model_type: str
    trained_at: float
    metrics: dict[str, float]
    git_sha: str
    threshold: float


class ErrorResponse(BaseModel):
    detail: str
