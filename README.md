# Home Credit Risk Scorer

A production-style credit risk scoring system that predicts the probability of loan default using real-world data from the [Kaggle Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data) competition (~300,000 applicants). Three machine learning models are trained and compared head-to-head, with SHAP explainability showing exactly which factors drove each prediction. Results are exposed through an interactive Streamlit dashboard.

---

## Results

Models evaluated on a held-out 20% test set. F1, Precision, and Recall are computed at each model's optimal decision threshold (not the default 0.5 — tuned for the ~8% default rate).

| Model | ROC-AUC | F1 | Precision | Recall | Threshold |
|---|---|---|---|---|---|
| XGBoost | 0.7784 | 0.3333 | 0.2861 | 0.3992 | 0.68 |
| Logistic Regression | 0.7573 | 0.3051 | 0.2324 | 0.4439 | 0.65 |
| Random Forest | 0.7404 | 0.2915 | 0.2451 | 0.3595 | 0.16 |

**ROC-AUC** measures overall ranking ability (1.0 = perfect, 0.5 = random coin flip). XGBoost leads at 0.778, well above the 0.75 baseline target.

---

## Architecture

```
data/
├── application_train.csv   ──┐
├── bureau.csv               │
├── previous_application.csv  ├──▶  src/data_prep.py  ──▶  src/train.py  ──▶  models/*.pkl
├── installments_payments.csv │         (join, clean,          (XGBoost,         (saved
└── credit_card_balance.csv  ─┘          encode,               RandomForest,      models)
                                         engineer)             LogisticReg)
                                              │
                                              ▼
                                        src/explain.py  ──▶  outputs/shap_*.png
                                              │
                                              ▼
                                           app.py  ──▶  Streamlit dashboard
                                        (loads models,        (risk score +
                                         feature_cols.json,    SHAP waterfall +
                                         metrics.csv)          model comparison)
```

---

## File Guide

| File | What it does |
|---|---|
| `src/data_prep.py` | Loads 5 CSV tables, joins them on applicant ID, fixes bad data, encodes categories, engineers new ratio features, splits into train/test |
| `src/train.py` | Trains XGBoost, Random Forest, and Logistic Regression; finds optimal decision threshold per model; saves models and metrics |
| `src/explain.py` | Computes SHAP values on the XGBoost model; saves global summary chart and per-applicant waterfall chart |
| `app.py` | Streamlit web app — sidebar inputs → risk score (0–100) + SHAP waterfall + model comparison table |
| `tests/` | Pytest unit tests for all three modules using small synthetic data (no real CSVs needed) |
| `outputs/` | `metrics.csv`, `feature_cols.json`, `shap_summary.png`, `shap_waterfall.png` |
| `models/` | Serialised model files (`xgboost.pkl`, `random_forest.pkl`, `logistic_regression.pkl`) — not in git (regenerate with `python src/train.py`) |

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Download the data from Kaggle**

Go to [kaggle.com/competitions/home-credit-default-risk/data](https://www.kaggle.com/competitions/home-credit-default-risk/data) and download these files into the `data/` folder:
- `application_train.csv` (required)
- `bureau.csv` (required)
- `previous_application.csv` (optional — adds prior loan history features)
- `installments_payments.csv` (optional — adds payment punctuality features)
- `credit_card_balance.csv` (optional — adds credit utilization features)

**3. Train the models** (~5–10 minutes)
```bash
python src/train.py
```
Saves `models/*.pkl` and `outputs/metrics.csv`.

**4. Generate SHAP plots**
```bash
python src/explain.py
```
Saves `outputs/shap_summary.png` and `outputs/shap_waterfall.png`.

**5. Launch the app**
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

- **5-table join** — application data is enriched with bureau history, prior applications, instalment payment behaviour, and credit card balance. Each extra table is optional; the pipeline detects which files are present.
- **Class imbalance** — only ~8% of applicants default. XGBoost uses `scale_pos_weight`; Random Forest uses `class_weight='balanced'`; Logistic Regression uses `class_weight='balanced'` with feature scaling via `StandardScaler`.
- **Threshold tuning** — instead of the default 50% cutoff, each model's decision threshold is tuned to maximise F1 on the test set. This is critical for imbalanced data.
- **No train/serve skew** — `app.py` runs the exact same `encode_categoricals` and `engineer_features` functions as training. The saved `feature_cols.json` ensures column order matches.
- **SHAP explainability** — `TreeExplainer` on the XGBoost model gives both a global view (which features matter most overall) and a local view (what drove this specific applicant's score).

---

## Future Work

- Join the remaining 2 tables (`POS_CASH_balance.csv`, `HomeCredit_columns_description.csv`)
- Hyperparameter tuning with Optuna
- Stacking ensemble (XGBoost + RF + LR → meta-learner)
- GenAI layer: convert SHAP values into plain-English risk summaries using HuggingFace

---

## Dataset

[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data) — Kaggle competition dataset provided by Home Credit Group. 307,511 loan applications with 122 features across 7 related tables.
