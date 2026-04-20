# Home Credit Risk Scorer

A production-style credit risk scoring system built on the [Kaggle Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data) dataset (~300,000 applicants). Three ML models are trained and compared head-to-head. A SHAP explainability layer shows exactly which factors drove each prediction. A business rules engine enforces hard policy caps on top of the model output — mirroring how real credit scoring systems work at lenders and bureaus.

Results are exposed through an interactive Streamlit dashboard where you can adjust any applicant input and see the score, explanation, and any triggered policy rules update instantly.

---

## Results

Evaluated on a held-out 20% test set. Thresholds are tuned per model to maximise F1 on the ~8% default rate — not left at the default 0.5 cutoff.

| Model | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| XGBoost | 0.7784 | 0.3333 | 0.2861 | 0.3992 |
| Logistic Regression | 0.7573 | 0.3051 | 0.2324 | 0.4439 |
| Random Forest | 0.7404 | 0.2915 | 0.2451 | 0.3595 |

ROC-AUC of 0.778 means the model correctly ranks a random defaulter as riskier than a random non-defaulter 77.8% of the time. Top Kaggle scores on this competition are ~0.80.

---

## Architecture

```
data/
├── application_train.csv  ─┐
├── bureau.csv              │
├── previous_application.csv├──▶ src/data_prep.py ──▶ src/train.py ──▶ models/*.pkl
├── installments_payments.csv│      (join, clean,       (XGBoost,
└── credit_card_balance.csv ─┘       encode, engineer)   RF, LR)
                                           │
                                           ▼
                                    src/explain.py ──▶ outputs/shap_*.png
                                           │
                                           ▼
                                        app.py ──▶ Streamlit dashboard
                                   (model score +        (risk score,
                                    guardrails layer +    SHAP waterfall,
                                    SHAP waterfall)       policy alerts)
```

---

## File Guide

| File | What it does |
|---|---|
| `src/data_prep.py` | Loads up to 5 CSV tables, joins on applicant ID, cleans anomalies, encodes categoricals, engineers ratio and interaction features, splits train/test |
| `src/train.py` | Trains XGBoost, Random Forest, Logistic Regression; tunes decision thresholds; saves model files and metrics |
| `src/explain.py` | Computes SHAP values on XGBoost; saves global summary and per-applicant waterfall charts |
| `app.py` | Streamlit app — sidebar inputs → model score → guardrails layer → SHAP explanation + policy alerts |
| `tests/` | Pytest unit tests for all modules using synthetic data (no real CSVs required) |
| `outputs/` | `metrics.csv`, `feature_cols.json`, SHAP charts |
| `models/` | Serialised model files — excluded from git, regenerate with `python src/train.py` |
| `docs/concepts.md` | Deep-dive reference covering every algorithm, metric, and design decision used |

---

## Two-Layer Scoring System

**Layer 1 — ML model (XGBoost)**
Trained on 300K applicants. Key feature groups:
- External credit scores (`EXT_SOURCE_1/2/3`) — strongest individual predictors
- Interaction features (`EXT_MEAN × ANNUITY_INCOME_RATIO`, `DEBT_STRESS`) — teach the model that a good credit score does not override an impossible repayment burden
- Bureau history — overdue days, active credit lines, total debt
- Financial ratios — loan-to-income, repayment-to-income, implied loan term

**Layer 2 — Business rules engine**
Hard policy caps applied after the model score, matching how production credit systems at lenders and bureaus work:

| Rule | Condition | Action |
|---|---|---|
| Repayments exceed income | `annuity / income > 1.0` | Cap score at 30 |
| Severe repayment burden | `annuity / income > 0.5` | Cap score at 55 |
| Extreme loan-to-income | `loan / income > 10×` | Cap score at 45 |
| Serious overdue history | `bureau_max_overdue > 60 days` | Reduce score by 15 |

The UI shows the raw model score and the policy-adjusted score separately, with the triggered rules listed explicitly — making the system auditable.

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Download data from Kaggle**

[kaggle.com/competitions/home-credit-default-risk/data](https://www.kaggle.com/competitions/home-credit-default-risk/data) — download into `data/`:
- `application_train.csv` (required)
- `bureau.csv` (required)
- `previous_application.csv` (optional)
- `installments_payments.csv` (optional)
- `credit_card_balance.csv` (optional)

**3. Train the models**
```bash
python src/train.py
```

**4. Generate SHAP charts**
```bash
python src/explain.py
```

**5. Launch the dashboard**
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`.

**6. Run tests**
```bash
python -m pytest tests/ -v
```

---

## Key Design Decisions

- **Train/serve consistency** — `app.py` runs the exact same `encode_categoricals` and `engineer_features` functions used at training time. `feature_cols.json` ensures column order never drifts.
- **Class imbalance** — 8% default rate handled with `scale_pos_weight` (XGBoost), `class_weight='balanced'` (RF, LR), and stratified splitting — not SMOTE.
- **Threshold tuning** — each model's decision threshold is tuned to maximise F1 rather than using the default 0.5, which is meaningless for imbalanced data.
- **Interaction features** — `EXT_MEAN × ANNUITY_INCOME_RATIO` and `DEBT_STRESS` were added after discovering the base model scored a borrower with 267% debt-to-income as Low Risk purely because external scores were high.
- **Explainability** — `TreeExplainer` SHAP gives both a global view (which features matter most across all applicants) and a local view (what drove this specific score). Required in production credit systems for adverse action notices.

---

## Dataset

[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data) — provided by Home Credit Group. 307,511 loan applications across 7 related tables.
