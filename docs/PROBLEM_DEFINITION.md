# Problem Definition & Requirements

## 1. Business context
A mid-size telecom operator loses ~26.5 % of its customers per year (Telco dataset base rate). Acquiring a customer costs
5–7× more than retaining one, so the retention team wants a **ranked list of at-risk customers every week** plus a
**reason per customer** so agents can pick the right offer (e.g. move a month-to-month fibre customer to a 1-year contract).

**Problem statement.** Predict, for every active customer, the probability of churning in the next billing cycle, explain the
main drivers, and keep the prediction service reliable and trustworthy as the customer base evolves.

## 2. Users & use cases
| Persona | Use case |
|---|---|
| Retention agent | Look up one customer → probability, risk band, top-5 reasons (`/v1/explain`) |
| Marketing analyst | Score a segment in bulk (`/v1/predict/batch`, ≤ 500) and target top decile |
| ML engineer | Retrain, compare experiments (MLflow), promote through the gate, roll back |
| On-call engineer | Get a Telegram alert when the service, model or data degrades |
| Compliance officer | Review fairness report, model card, data-minimisation choices |

## 3. Requirements (MoSCoW)
**Functional** — M: single & batch prediction; M: explanation per prediction; M: automated weekly retraining; M: drift detection with alert;
M: model versioning/promotion gate; S: hot-reload of a new model; S: HTML drift report; C: auto-retrain on drift (implemented).

**Non-functional** — M: p95 latency < 500 ms (alert threshold) ; M: availability signal + alert; M: reproducible runs (pinned deps, params in git);
M: no secrets in git; S: ≥ 80 % test coverage (achieved 94 %); S: non-root containers; C: HPA 2–6 pods.

**Out of scope** — real-time feature store, label feedback loop UI, multi-region deployment, automated price/offer optimisation.

## 4. Success metrics
| Level | Metric | Target | Measured |
|---|---|---|---|
| Business | Precision in the top-10 % risk decile | ≥ 2× base rate (≥ 53 %) | **74.3 %** (lift 2.8×; base rate 26.5 %) |
| Business | Share of all churners captured in top-20 % | ≥ 40 % | **50.8 %** (precision 67.9 %) |
| Business | Recall of churners at default threshold | ≥ 0.75 | **0.80** |
| Model | ROC-AUC (hold-out) | ≥ 0.80 (gate 0.78) | **0.847** |
| Model | PR-AUC | ≥ 0.60 | **0.664** |
| Model | Fairness: disparate-impact ratio | ≥ 0.80 (4/5 rule) | gender **0.99** ✔ · SeniorCitizen **0.57** ✘ (see RESPONSIBLE_AI) |
| System | p95 latency `/v1/*` | < 500 ms | alert-enforced; single inference < 100 ms (test) |
| System | 5xx error ratio | < 5 % (alert) | alert-enforced |
| System | Time from drift to alert | < 1 h | hourly DAG |
| System | Test coverage | ≥ 80 % | **94 %** |

_Business metrics are computed on the hold-out set (1 409 customers); the validation gate enforces model metrics only (AUC, recall)._

## 5. Constraints & assumptions
Public dataset (static, 7 043 rows) stands in for a customer warehouse; labels (actual churn) are not available in real time,
so production monitoring is limited to **data and prediction drift** — performance drift needs delayed labels (future work).
Free-tier tooling only (GitHub Actions, GHCR, open-source stack). The model never receives `gender`.
