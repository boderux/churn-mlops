# Presentation plan (15–20 min + 10 min Q&A) — all 4 members speak and drive the demo

| # | Min | Slide(s) | Speaker | Key content |
|---|---|---|---|---|
| 1 | 2 | Problem & business case | M1 | 26.5 % churn, cost of acquisition vs retention, who uses the system, success metrics (AUC 0.847, recall 0.80, top-decile lift 2.8×) |
| 2 | 3 | Architecture & decisions | M2 | Diagram (README), 10-service stack, 3 trade-offs: venv-isolated Airflow, model baked-in vs mounted, GitOps vs push CD |
| 3 | 4 | ML pipeline deep dive | M1 | Validation rules, features in-pipeline (no train/serve skew), 3 models + CV search, MLflow, **validation gate** |
| 4 | 3 | Serving & monitoring | M2 → M3 | API design, Prometheus metrics, Grafana, 10 alerts, Telegram |
| 5 | 3 | Airflow DAGs + Evidently | M3 | DAG graphs, drift → alert → retrain loop, threshold choice (16 % vs 37 %), limits (no labels) |
| 6 | 3 | CI/CD + ArgoCD | M4 | CI jobs, CD bump-manifest, dev auto / prod tag, rollback via git revert |
| 7 | 3 | Responsible AI | M4 | SHAP + LIME agree, **seniors fail the 4/5 rule (0.57)**, mitigation trade-off, why we did not ship group thresholds, privacy, ethics |
| 8 | – | Live demo | all | see script |
| 9 | 1 | Lessons & future work | M1 | Label feedback loop, performance drift, feature store, canary deploys |

## Live demo script (≈ 5 min inside the slots above — rehearse twice; keep a recorded fallback video)
1. **M2** – `docker compose ps` (all healthy) → Swagger → `/v1/predict` and `/v1/explain` with the example customer (SHAP top drivers).
2. **M3** – Grafana dashboard live; run `make traffic` → panels move. Airflow UI: show both DAGs' graph views.
3. **M3** – `make traffic-drift` → trigger/await `churn_drift_monitoring` → **Telegram alert on the projector phone/Telegram Web** → drift report HTML (drifted columns) → training DAG auto-triggers.
4. **M1** – MLflow: nested runs per model; show the gate decision / "new champion" Telegram message; API `reload` → `/v1/model` shows new version.
5. **M4** – GitHub Actions green run → `git commit` bump in `k8s/overlays/dev` → ArgoCD UI shows OutOfSync → Synced/Healthy + Telegram message; show `git revert` rollback idea.
6. **M4** – open `fairness_report.json` / SHAP plot.

## Anticipated Q&A (assign an owner per question)
* *Why is XGBoost chosen if LogReg is within 0.001 AUC?* (M1) CV-AUC selection + gate; we'd choose LogReg if interpretability or latency dominated; difference is within noise.
* *Why isn't the 0.5 threshold optimised?* (M1) Business cost matrix unknown; threshold is a parameter (`metadata.threshold`); recall already 0.80. Future: cost-based threshold.
* *Evidently flags drift on healthy data?* (M3) KS tests on 300 rows are noisy → share-of-columns rule (30 %), healthy ≈ 16 %, shifted ≈ 37 %; tune with a larger window.
* *Can you detect performance degradation?* (M3) Not without labels; only data/prediction drift. Plan: join delayed churn labels, compute rolling AUC.
* *Why Pushgateway?* (M3) Batch DAG isn't scrapeable; staleness handled by `DriftMonitorStale`.
* *Why GitOps?* (M4) No cluster creds in CI, audit trail, self-heal, rollback by revert.
* *Is the prod rollout safe?* (M4) RollingUpdate maxUnavailable 0, readiness probe requires a loaded model, PDB, HPA.
* *Fairness: why not deploy the mitigated model?* (M4) Still < 0.8 DI, lower accuracy, group thresholds = disparate-treatment risk; part of the gap is real base-rate difference; decision documented.
* *What if the model is wrong for a customer?* (M2/M4) Scores are advisory for retention outreach only; explanations + human review.
