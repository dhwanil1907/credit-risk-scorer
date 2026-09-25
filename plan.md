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

- [ ] Pydantic request model with validation and sensible error messages on bad input
- [ ] Response: `{score, band, rules_applied[], top_reasons[3]}`
- [ ] SHAP reasons use the plain-English labels already in `app.py`'s `FEATURE_LABELS`
- [ ] `GET /health` for load balancer checks
- [ ] Build the `TreeExplainer` once at startup, not per request — it is expensive and will
      dominate p95 latency otherwise
- [ ] API tests: valid request, malformed payload, out-of-range values, missing fields

## Day 3 — Docker

- [ ] Multi-stage Dockerfile, non-root user, XGBoost artifact only
- [ ] `docker-compose.yml` to run API + Streamlit together locally
- [ ] Point Streamlit at the API instead of loading the model itself, so there is one scoring path
- [ ] Verify the same image runs locally and on AWS without changes

## Day 4 — CI pipeline

- [ ] `.github/workflows/ci.yml`: 24 unit tests, `ruff` lint, Docker build on every push
- [ ] Branch protection so merges are blocked when any step fails
- [ ] pip caching to keep runs fast
- [ ] CI badge in `README.md`

## Day 5 — AWS deployment

- [ ] Push image to ECR
- [ ] Deploy to App Runner (less setup) or EC2 (more control, more to explain)
- [ ] Config and secrets via environment variables — nothing hardcoded, nothing committed
- [ ] Load test for p50/p95 latency; record numbers in the README
- [ ] Extend CI to build and push on merge to `main`

## Day 6 — MLflow versioning

- [ ] Log params, metrics, and artifacts for all three models from `src/train.py`
- [ ] Register XGBoost in the model registry with stages
- [ ] Version the training data — hash the input CSVs and log the hash as a run tag, since 1.8GB
      cannot go in git
- [ ] API loads a specific registered version, not a loose file path

## Day 7 — Drift monitoring and retraining trigger

**PSI is written by hand, not pulled from a library**, so it can be explained in interviews.

- [ ] Persist a reference distribution from the training set (bin edges + proportions per feature)
- [ ] PSI implementation with unit tests: identical distributions score ~0, known shifts score as
      expected, and empty bins do not divide by zero
- [ ] Replay held-out test data in batches as the simulated incoming feed
- [ ] Simulate realistic shifts: incomes down 15%, default rate rising, bureau scores degrading
- [ ] Flag features above PSI 0.2; record the smallest shift the detector reliably catches
- [ ] Retraining trigger script: retrain, evaluate against the current registered model, promote
      only if it performs better on the shifted data

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
