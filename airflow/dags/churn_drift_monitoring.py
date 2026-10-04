"""DAG 2 - churn_drift_monitoring (Evidently)

Every hour: compare the production prediction log (current window) with the champion's
reference data using Evidently's DataDriftPreset (+ prediction drift on churn_probability).

   check_drift (Evidently report -> Pushgateway -> Prometheus/Grafana)
       --branch--> drift:    Telegram alert (inside CLI) + trigger churn_training_pipeline
       \\--------> no drift:  nothing to do
"""

from __future__ import annotations

import json
import os
from datetime import timedelta

import pendulum

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

PROJECT = os.getenv("PROJECT_ROOT", "/opt/project")
PY = os.getenv("ML_PYTHON", "/opt/airflow/venv/bin/python")
ENV = {
    "PYTHONPATH": f"{PROJECT}/src",
    "PROJECT_ROOT": PROJECT,
    "PARAMS_PATH": f"{PROJECT}/configs/params.yaml",
}


def telegram_failure(context: dict) -> None:
    import sys

    sys.path.insert(0, f"{PROJECT}/src")
    from churn.notify import send_telegram

    ti = context["task_instance"]
    send_telegram(f"Drift monitoring failed: <code>{ti.task_id}</code>\n{ti.log_url}", "critical")


def decide(ti, **_: object) -> str:
    raw = ti.xcom_pull(task_ids="check_drift") or "{}"
    try:
        result = json.loads(raw.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return "no_drift"
    return "trigger_retraining" if result.get("dataset_drift") else "no_drift"


with DAG(
    dag_id="churn_drift_monitoring",
    description="Hourly Evidently drift check with Telegram alerting and auto-retraining",
    schedule="@hourly",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "mlops-team",
        "retries": 1,
        "retry_delay": timedelta(minutes=1),
        "on_failure_callback": telegram_failure,
    },
    tags=["churn", "monitoring", "evidently"],
    doc_md=__doc__,
) as dag:
    check = BashOperator(
        task_id="check_drift",
        cwd=PROJECT,
        env=ENV,
        append_env=True,
        bash_command=f"{PY} -m churn.cli drift 2>/dev/null | tail -n 1",  # last stdout line = JSON -> XCom
    )
    branch = BranchPythonOperator(task_id="branch_on_drift", python_callable=decide)
    retrain = TriggerDagRunOperator(
        task_id="trigger_retraining",
        trigger_dag_id="churn_training_pipeline",
        wait_for_completion=False,
        reset_dag_run=True,
    )
    no_drift = EmptyOperator(task_id="no_drift")

    check >> branch >> [retrain, no_drift]
