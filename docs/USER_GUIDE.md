# User & Operations Guide

## 1. Prerequisites
Docker 24+ with Compose v2, Python 3.11 (for local runs), `make`. For K8s: a cluster, `kubectl`, ArgoCD ≥ 2.9.

## 2. Configuration
| What | Where |
|---|---|
| Pipeline parameters, thresholds, gate, drift threshold | `configs/params.yaml` (override file with `PARAMS_PATH`) |
| Secrets & ports | `.env` (copy from `.env.example`) |
| Alert thresholds | `monitoring/alert_rules.yml` |
| Telegram receiver | `monitoring/alertmanager.yml` (token/chat id rendered from env at start) |

## 3. Day-1 runbook
1. `cp .env.example .env`, fill Telegram values (optional), set `ADMIN_TOKEN`.
2. `make up` then `make bootstrap` (or in Airflow UI → enable & trigger **churn_training_pipeline**).
3. Check `http://localhost:8000/ready` → `ready`; open Grafana dashboard.
4. Enable **churn_drift_monitoring** in Airflow (hourly). Generate traffic: `make traffic`.
5. Send a test Telegram message: `python -c "from churn.notify import send_telegram as s; s('hello','ok')"` (with env set, `PYTHONPATH=src`).

## 4. Operations
| Task | How |
|---|---|
| Retrain now | Airflow → trigger `churn_training_pipeline` or `make pipeline` |
| Why was a model rejected? | Telegram message / `artifacts/gate_decision.json` |
| Compare experiments | MLflow UI :5000 → experiment `churn-prediction` (nested run per model) |
| Roll back the model (Compose) | Restore previous `artifacts/champion` bundle, `POST /v1/admin/reload -H 'X-Admin-Token: …'` |
| Roll back a deployment (K8s) | `git revert` the CD bump commit → ArgoCD syncs back |
| Inspect drift | `artifacts/reports/drift_report.html`; Grafana panel *Evidently drift share* |
| Tune sensitivity | `monitoring.drift_share_threshold` (default 0.30). Healthy windows showed ≈ 16 % drift by chance — keep margin |
| Silence an alert | Alertmanager UI :9093 → Silences |

## 5. Alert catalogue
| Alert | Severity | Meaning → first action |
|---|---|---|
| ChurnApiDown | critical | Scrape failing → `docker compose ps`, `logs api` |
| ModelNotLoaded | critical | No champion → run training DAG |
| HighErrorRate | critical | 5xx > 5 % for 5 m → check API logs/inputs |
| HighLatencyP95 | warning | p95 > 0.5 s → CPU/HPA, batch size |
| DataDriftDetected / SevereDataDrift | warning / critical | Evidently flagged drift → inspect report; retraining auto-triggered |
| DriftMonitorStale | warning | Drift DAG not running → Airflow scheduler |
| ChurnPredictionRateAnomaly | warning | > 60 % predicted churn → prediction drift / upstream data bug |
| LowModelRocAuc | warning | Deployed AUC < 0.75 |
| ModelStale | warning | Model > 14 days old → weekly DAG failing |

## 6. Kubernetes + ArgoCD setup
```bash
# 1. image registry: GitHub → Packages (ghcr.io/<org>/churn-api); make it public or add an imagePullSecret
# 2. secrets
kubectl create ns churn && kubectl -n churn create secret generic churn-api-secrets --from-literal=admin-token=$(openssl rand -hex 24)
kubectl -n argocd create secret generic argocd-notifications-secret --from-literal=telegram-token=<BOT_TOKEN>
# 3. ArgoCD objects
kubectl apply -n argocd -f argocd/project.yaml -f argocd/application-dev.yaml -f argocd/application-prod.yaml -f argocd/notifications-cm.yaml
# 4. GitHub → Settings → Secrets: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID; Actions → Workflow permissions: read & write
```
Deploy dev: merge to `main`. Deploy prod: `git tag v1.0.0 && git push --tags`.

## 7. Development
`make lint typecheck test` · tests are marked `slow` when they train models (`pytest -m "not slow"` for a 2 s loop).
Add a feature: edit `features.py` → tests → `make pipeline`. Add a model: add an entry in `train.get_search_spaces`.
