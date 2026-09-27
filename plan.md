# Credit Risk Scorer — Production Deployment

Take the existing Home Credit Risk Scorer from a Streamlit demo to a deployed, tested, monitored
service. The model is done (XGBoost, 0.778 ROC-AUC); this phase is all infrastructure.

**Timeline:** ~1 week &nbsp;•&nbsp; **Targets:** DS Intern, MLE Intern
**Gap filled:** Docker, hands-on AWS, CI/CD, model monitoring
**Stack:** Python, XGBoost, SHAP, FastAPI, Docker, AWS (EC2 or App Runner), GitHub Actions, MLflow

## Current state

| Thing | Status |
|---|---|
| Models | XGBoost (2.5MB), Random Forest (323MB), Logistic Regression (4.6KB) in `models/` — gitignored |
| Tests | 24 passing (`test_data_prep.py` 10, `test_guardrails.py` 7, `test_train.py` 4, `test_explain.py` 3) |
| Data | All 5 Kaggle CSVs in `data/` (1.8GB), gitignored |
| Serving | Streamlit (`app.py`), local only |
| Python | 3.13 in `.venv` (SHAP has no 3.14 wheels yet) |
| Traffic | No real users — drift input is replayed held-out data, labelled as simulated |

## Known issues to fix along the way

- `app.py` still has `_align_to_model` for backward compatibility with older pickles; fresh
  training (Day 1) makes feature names match `feature_cols.json` (62 columns).
- Random Forest at 323MB cannot go in a container image. Serve XGBoost only.

---

## Day 1 — Retrain and restructure

- [x] Retrain on the full 5-table dataset so the artifact matches `feature_cols.json`
- [x] Move `apply_guardrails` from `app.py` to `src/guardrails.py`; update `tests/test_guardrails.py`
- [x] Extract SHAP helpers into `src/shap_utils.py` (`coerce_shap_matrix`, `expected_value_scalar`)
- [x] Confirm all 24 tests still pass

## Day 2 — Scoring API

`POST /predict` returns risk score, risk band, business-rule caps applied, and the top 3 SHAP
reasons for that applicant.

- [x] Pydantic request model with validation and sensible error messages on bad input
- [x] Response: `{score, band, rules_applied[], top_reasons[3]}`
- [x] SHAP reasons use the plain-English labels in `src/labels.py`
- [x] `GET /health` for load balancer checks
- [x] `Scorer` builds the `TreeExplainer` once at startup
- [x] API tests: valid request, malformed payload, out-of-range values, missing fields (30 total)

## Day 3 — Docker

- [x] Multi-stage Dockerfile, non-root user, XGBoost artifact only
- [x] `docker-compose.yml` to run API + Streamlit together locally
- [x] Streamlit calls `SCORER_API_URL` (default `http://127.0.0.1:8000`) instead of loading a model
- [x] Image built and running locally via Colima (`docker compose up --build`)

## Day 4 — CI pipeline

- [x] `.github/workflows/ci.yml`: pytest, ruff, Docker build on every push
- [ ] Branch protection (GitHub setting, after the workflow is on `main`): require the CI check before merge
- [x] pip caching
- [x] CI badge in `README.md`

AWS, MLflow, and drift monitoring are out of scope.

---

## How it gets evaluated

| Measure | Target |
|---|---|
| API latency | p50 and p95 under a simple load test, numbers published in the README |
| Drift sensitivity | Which simulated shifts are caught, and at what magnitude |
| Retraining value | Whether the retrained model beats the original on the shifted data |

## Deliverables

- [ ] Live API endpoint
- [ ] Architecture diagram
- [ ] Drift report
- [ ] README with a decisions log

## Numbers that are true and worth quoting

307K applications • 5 joined tables • 0.778 ROC-AUC • **24 unit tests** •
267% debt-to-income as the finding that motivated the interaction features
