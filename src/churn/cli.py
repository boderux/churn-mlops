"""Command line entry-point used by Airflow, Make, CI and humans.

    python -m churn.cli <ingest|validate|preprocess|train|gate|promote|fairness|explain|drift|run-all>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from churn import data, drift, evaluate, explain, fairness, model_io, storage, train
from churn.config import ARTIFACT_DIR, CHAMPION_DIR, load_params
from churn.notify import send_telegram

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("churn.cli")


def cmd_validate(params) -> int:
    errors = data.validate(data.load_raw(params), params["data"]["target"])
    if errors:
        send_telegram("Data validation FAILED:\n" + "\n".join(f"• {e}" for e in errors), "critical")
        log.error("Validation errors: %s", errors)
        return 1
    log.info("Data validation passed")
    return 0


def cmd_promote(params) -> int:
    decision = json.loads((ARTIFACT_DIR / "gate_decision.json").read_text())
    if not decision["promote"]:
        log.warning("Candidate rejected: %s", decision["reasons"])
        send_telegram(f"Candidate <b>{decision['model_type']}</b> rejected:\n"
                      + "\n".join(f"• {r}" for r in decision["reasons"]), "warning")
        return 0
    model_io.promote(train.CANDIDATE_DIR, CHAMPION_DIR)
    storage.sync_artifacts_to_minio()
    send_telegram(
        f"New champion <b>{decision['model_type']}</b> v{decision['version']}\n"
        f"ROC-AUC {decision['candidate_auc']:.3f} (prev: {decision['champion_auc']})", "ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="churn")
    ap.add_argument("command", choices=["ingest", "validate", "preprocess", "train", "gate",
                                        "promote", "fairness", "explain", "drift", "run-all",
                                        "sync-data", "sync-reports", "sync-artifacts", "sync-all"])
    ap.add_argument("--quick", action="store_true", help="fast training (CI / smoke tests)")
    ap.add_argument("--params", default=None)
    args = ap.parse_args(argv)
    params = load_params(args.params)
    c = args.command
    if c == "ingest":
        data.ingest(params)
    elif c == "validate":
        return cmd_validate(params)
    elif c == "preprocess":
        data.preprocess(params)
    elif c == "train":
        train.train(params, quick=args.quick)
    elif c == "gate":
        evaluate.gate(params)
    elif c == "promote":
        return cmd_promote(params)
    elif c == "fairness":
        fairness.run(params)
    elif c == "explain":
        explain.run(params)
    elif c == "drift":
        res = drift.run(params)
        drift.alert_if_drift(res)
        print(json.dumps(res))
        return 0
    elif c == "sync-data":
        storage.sync_data_to_minio(params)
    elif c == "sync-reports":
        storage.sync_reports_to_minio()
    elif c == "sync-artifacts":
        storage.sync_artifacts_to_minio()
    elif c == "sync-all":
        storage.sync_all_to_minio(params)
    elif c == "run-all":
        data.ingest(params)
        if cmd_validate(params):
            return 1
        data.preprocess(params)
        train.train(params, quick=args.quick)
        evaluate.gate(params)
        cmd_promote(params)
        fairness.run(params)
        explain.run(params)
        storage.sync_all_to_minio(params)
    return 0


if __name__ == "__main__":
    sys.exit(main())
