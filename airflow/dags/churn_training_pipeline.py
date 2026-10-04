"""DAG 1 - churn_training_pipeline

ingest -> validate -> preprocess -> train (MLflow) -> gate -> branch:
    gate passed -> promote + hot-reload API   |   gate failed -> reject (Telegram)
then: fairness audit -> SHAP/LIME report -> Telegram summary

Schedule: weekly, plus on-demand (triggered by the drift-monitoring DAG).
Heavy ML dependencies live in an isolated virtualenv (/opt/airflow/venv) so they never
clash with Airflow's own constraints; tasks call the project CLI (`python -m churn.cli`).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import timedelta
from pathlib import Path

import pendulum

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

PROJECT = os.getenv("PROJECT_ROOT", "/opt/project")
PY = os.getenv("ML_PYTHON", "/opt/airflow/venv/bin/python")
sys.path.insert(0, f"{PROJECT}/src")

ENV = {
    "PYTHONPATH": f"{PROJECT}/src",
    "PROJECT_ROOT": PROJECT,
    "PARAMS_PATH": f"{PROJECT}/configs/params.yaml",
    "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
    "MLFLOW_S3_ENDPOINT_URL": os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000"),
    "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
    "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin"),
    "AWS_DEFAULT_REGION": os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
}


def telegram_failure(context: dict) -> None:
    """on_failure_callback: page the team on Telegram (Airflow env has `requests`)."""
    from churn.notify import send_telegram

    ti = context["task_instance"]
    send_telegram(
        f"Airflow task failed\nDAG: <code>{ti.dag_id}</code>\nTask: <code>{ti.task_id}</code>\n"
        f"Run: {context['run_id']}\n{ti.log_url}",
        level="critical",
    )


def cli(task_id: str, command: str, extra: str = "", **kw) -> BashOperator:
    return BashOperator(
        task_id=task_id,
        cwd=PROJECT,
        env=ENV,
        append_env=True,
        bash_command=f"{PY} -m churn.cli {command} {extra}",
        **kw,
    )


def choose(**_: object) -> str:
    decision = json.loads(
        Path(os.getenv("ARTIFACT_DIR", f"{PROJECT}/artifacts"), "gate_decision.json").read_text()
    )
    return "promote_model" if decision["promote"] else "reject_model"


with DAG(
    dag_id="churn_training_pipeline",
    description="End-to-end churn model (re)training with validation gate and Responsible-AI reports",
    schedule="@weekly",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "mlops-team",
        "retries": 1,
        "retry_delay": timedelta(minutes=2),
        "on_failure_callback": telegram_failure,
    },
    tags=["churn", "training", "mlops"],
    doc_md=__doc__,
) as dag:
    ingest = cli("ingest_data", "ingest")
    validate = cli("validate_data", "validate", retries=0)
    preprocess = cli("preprocess_split", "preprocess")
    train = cli("train_models", "train", execution_timeout=timedelta(hours=1))
    gate = cli("validation_gate", "gate")
    branch = BranchPythonOperator(task_id="branch_on_gate", python_callable=choose)
    # `promote` CLI replaces the champion bundle and notifies Telegram.
    promote = cli("promote_model", "promote")
    reload_api = BashOperator(
        task_id="reload_api",
        retries=3,
        bash_command='curl -fsS -X POST -H "X-Admin-Token: ${ADMIN_TOKEN}" ${API_URL:-http://api:8000}/v1/admin/reload',
    )
    reject = cli("reject_model", "promote")  # same CLI: sends the rejection reasons to Telegram
    fairness = cli("fairness_audit", "fairness", trigger_rule="none_failed_min_one_success")
    explain = cli("explainability_report", "explain")
    sync_storage = cli("sync_storage", "sync-all", trigger_rule="none_failed_min_one_success")
    done = PythonOperator(
        task_id="notify_done",
        python_callable=lambda: __import__("churn.notify", fromlist=["send_telegram"]).send_telegram(
            "Training pipeline finished. MinIO storage, Fairness + SHAP/LIME reports updated.", "ok"
        ),
    )
    skip = EmptyOperator(task_id="end")

    ingest >> validate >> preprocess >> train >> gate >> branch
    branch >> promote >> reload_api >> fairness
    branch >> reject >> fairness
    fairness >> explain >> sync_storage >> done >> skip
