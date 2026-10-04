# Team roles & contribution rules

> Replace the names below and keep this file current — it is part of the individual-contribution evidence.

| Member | Role | Owns (primary) | Must be able to explain in Q&A |
|---|---|---|---|
| **Member 1** — _name / student id_ | Data & ML Lead | `src/churn/{data,features,train,evaluate}.py`, `configs/params.yaml`, MLflow experiments, data-quality + model tests | Feature engineering, CV/tuning, why XGBoost vs LogReg, validation gate, imbalance handling |
| **Member 2** — _name / student id_ | Serving & Platform | `src/api/*`, `Dockerfile*`, `docker-compose.yml`, integration tests, API docs | API design, Prometheus metrics, multi-stage build, health checks, hot-reload |
| **Member 3** — _name / student id_ | Orchestration & Monitoring | `airflow/dags/*`, `src/churn/{drift,notify}.py`, `monitoring/*`, Telegram bot | DAG design, Evidently config/thresholds, alert rules, Grafana panels |
| **Member 4** — _name / student id_ | DevOps & Responsible AI | `.github/workflows/*`, `k8s/*`, `argocd/*`, `src/churn/{fairness,explain}.py`, `docs/RESPONSIBLE_AI.md` | CI/CD + GitOps flow, ArgoCD sync/rollback, fairness findings, SHAP/LIME, ethics |

Everyone: writes tests for their code, reviews at least one other member's PR, and contributes to the slides + live demo.

## Workflow
* Branches: `main` (protected, deploys to dev) ← `feature/<member>-<topic>`; releases are tags `vX.Y.Z` (deploy to prod).
* Commits: Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `ci:`), small and frequent — commit history is graded.
* PR checklist: `make lint typecheck test` green, docs updated, ≥ 1 reviewer.
* Never commit secrets, data or model binaries.
