# Responsible AI Report

Numbers below come from the pipeline run on the real Telco dataset (`artifacts/reports/*`, copies in `docs/samples/`).
Re-generate with `python -m churn.cli fairness` and `python -m churn.cli explain`.

## 1. Explainability

**Global (SHAP, mean |value| on 300 test customers)** — `docs/figures/shap_importance.png`, `shap_summary.png`

| Rank | Feature | mean abs SHAP |
|---|---|---|
| 1 | Contract = Month-to-month | 0.568 |
| 2 | tenure | 0.351 |
| 3 | InternetService = Fiber optic | 0.293 |
| 4 | OnlineSecurity = No | 0.209 |
| 5 | Contract = Two year | 0.172 |
| 6 | PaymentMethod = Electronic check | 0.164 |

Interpretation: short tenure, flexible contracts, fibre plans without security add-ons and e-check payers drive churn; long
contracts protect. This matches domain intuition (and is asserted by model tests: month-to-month > two-year, new > tenured).

**Local** — `/v1/explain` (SHAP) for any customer; LIME cross-check for the highest-risk customer
(`explainability.json`): LIME's top drivers (month-to-month, low tenure, fibre) agree with SHAP's ⇒ explanations are consistent
across two independent methods. Caveat: both explain the *model*, not causal effects — agents should not read them as proof that
changing one attribute will retain the customer.

## 2. Fairness (Fairlearn, hold-out set, threshold 0.5)

| Attribute | Group | Selection rate | TPR | FPR |
|---|---|---|---|---|
| gender | Female / Male | 0.416 / 0.420 | 0.782 / 0.823 | 0.273 / 0.285 |
| SeniorCitizen | 0 / **1** | 0.373 / **0.658** | 0.768 / 0.898 | 0.254 / **0.468** |

| Attribute | Disparate-impact ratio | Demographic-parity diff | Equalised-odds diff | 4/5 rule |
|---|---|---|---|---|
| gender | 0.992 | 0.003 | 0.041 | ✔ pass |
| SeniorCitizen | **0.568** | 0.284 | 0.214 | ✘ **fail** |

**Findings**
* Gender: no meaningful disparity, and the model does not use it as an input.
* Seniors are flagged as churners 1.8× more often and suffer **almost twice the false-positive rate** (0.47 vs 0.25) → they would
  receive retention offers they do not need (cost: margin) — and lower accuracy (0.69 vs 0.75).
* Part of the gap is **real**: seniors churn more in the data (higher base rate). So demographic parity alone is the wrong target; the
  false-positive gap is the actionable harm.

**Mitigation tried** — Fairlearn `ThresholdOptimizer` (post-processing, equalised odds, group-specific thresholds):

| SeniorCitizen | DI ratio | Equalised-odds diff | FPR (0 / 1) | Accuracy (0 / 1) |
|---|---|---|---|---|
| baseline | 0.568 | 0.214 | 0.254 / 0.468 | 0.752 / 0.694 |
| mitigated | 0.708 | 0.095 | 0.268 / 0.363 | 0.730 / 0.680 |

Equalised-odds difference more than halves and FPR gap narrows, but DI is still < 0.8 and overall accuracy falls ~1–2 points.
**Decision:** keep the baseline in production (no group-specific thresholds), *monitor* the senior FPR in the fairness report each
training run, and cap the cost of false positives by offering seniors low-cost, non-discount actions (e.g. service call).
Group-specific thresholds raise their own legal/ethical questions (disparate treatment), so adopting them needs a business/legal review.
Further options: re-weighting at training time, dropping `SeniorCitizen`, calibration per group.

## 3. Data privacy
* **Minimisation:** `gender` is not accepted by the API and not used by the model; `customerID` is dropped before training and never logged.
* **Prediction log** stores only model features + probability (no direct identifiers) — retention should be limited (e.g. 30 days) in production.
* Pseudonymous public data here; with real customers: GDPR/PDPA lawful basis, DPIA, access control on MLflow/Airflow/Grafana, encryption at rest.
* No secrets in git; Telegram messages contain only aggregates (no customer data).

## 4. Ethical implications & mitigations
| Risk | Mitigation |
|---|---|
| Unequal treatment of seniors (higher FPR) | Fairness audit every run, non-monetary actions for flagged seniors, human review |
| Using predictions to deny service / worse prices to likely churners (or the opposite: ignore "loyal" customers) | Policy: scores are for *retention outreach only*; no pricing/eligibility use |
| Over-trust in explanations | Document that SHAP/LIME are associative; show uncertainty (risk bands) |
| Drift silently degrades a model | Hourly Evidently check + Telegram alert + gated retraining |
| Automated retraining amplifies bias | Gate on AUC/recall; fairness report generated after each promotion and reviewed by a human |
| Customers unaware of profiling | Disclose in privacy notice; allow opt-out from retention marketing |

## 5. Model card (summary)
*Intended use:* weekly retention targeting for postpaid telecom customers. *Not for:* credit, pricing, eligibility, employment decisions.
*Training data:* IBM Telco churn (7 043 rows, one US-style telco, snapshot). *Metrics:* AUC 0.847, recall 0.80, precision 0.51 @0.5.
*Limitations:* single snapshot, no temporal validation, no labels in production, seniors over-flagged, calibration not verified.
