# Home Credit Risk Scorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production-grade credit risk scoring system using XGBoost, Random Forest, and Logistic Regression on the Kaggle Home Credit Default Risk dataset (300K rows, 2 joined tables), with SHAP explainability and a live Streamlit demo.

**Architecture:** Join `application_train.csv` with aggregated `bureau.csv` on `SK_ID_CURR`, engineer features, train three classifiers head-to-head on ROC-AUC/F1, layer SHAP on XGBoost for interpretability, expose everything via Streamlit. An optional Phase 6 adds a HuggingFace NLP layer that converts SHAP values into natural-language risk summaries.

**Tech Stack:** Python 3.10+, XGBoost, scikit-learn, SHAP, Streamlit, pandas, NumPy, matplotlib, joblib, HuggingFace Transformers (optional)

---

## File Structure

```
credit-risk-scorer/
├── data/
│   ├── application_train.csv    # Kaggle download — main table (~300K rows)
│   └── bureau.csv               # Kaggle download — credit bureau history
├── src/
│   ├── __init__.py
│   ├── data_prep.py             # Load, join, clean, encode, engineer, split
│   ├── train.py                 # Train XGBoost/RF/LR; save models + metrics
│   ├── explain.py               # SHAP global + local plots
│   └── llm_summary.py          # (Phase 6) HuggingFace NL risk summary
├── models/
│   ├── xgboost_model.pkl
│   ├── rf_model.pkl
│   └── lr_model.pkl
├── outputs/
│   ├── feature_cols.json        # Saved feature column list for app inference
│   ├── shap_summary.png
│   ├── shap_waterfall.png
│   └── metrics.csv
├── tests/
│   ├── test_data_prep.py
│   ├── test_train.py
│   └── test_explain.py
├── app.py                       # Streamlit UI
├── requirements.txt
└── README.md
```

---

## Task 1: Project Setup & Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`

- [ ] **Step 1: Create requirements.txt** with pinned versions for: xgboost, scikit-learn, shap, streamlit, pandas, numpy, matplotlib, seaborn, joblib.

- [ ] **Step 2: Install dependencies** via `pip install -r requirements.txt`. Verify no errors.

- [ ] **Step 3: Create `src/__init__.py`** as an empty file to make `src` a package.

- [ ] **Step 4: Verify dataset files exist** by loading the first 5 rows of each CSV and printing their shapes. `application_train.csv` should have 122 columns; `bureau.csv` should have 17 columns.

- [ ] **Step 5: Commit** with message `feat: project setup and dependencies`.

---

## Task 2: Data Loading, Joining & Cleaning

**Files:**
- Create: `src/data_prep.py`
- Create: `tests/test_data_prep.py`

### What `data_prep.py` must do

**`load_bureau_aggregates(bureau_df)`**
Group `bureau.csv` by `SK_ID_CURR`. Produce one row per applicant with these aggregates: `bureau_count` (count of records), `bureau_active_count` (count where `CREDIT_ACTIVE == 'Active'`), `bureau_max_overdue` (max of `CREDIT_DAY_OVERDUE`), `bureau_total_debt` (sum of `AMT_CREDIT_SUM_DEBT`), `bureau_avg_credit` (mean of `AMT_CREDIT_SUM`).

**`clean_application(df)`**
- Replace `DAYS_EMPLOYED == 365243` with `NaN` — this is a known anomalous placeholder for unemployed/pensioners, not a real value.
- Impute all remaining numeric nulls with the column median.

**`encode_categoricals(df)`**
Label-encode four columns with explicit mappings:
- `CODE_GENDER`: M→1, F→0, XNA→0
- `NAME_CONTRACT_TYPE`: Cash loans→1, Revolving loans→0
- `FLAG_OWN_CAR`: Y→1, N→0
- `FLAG_OWN_REALTY`: Y→1, N→0

Drop any remaining object-dtype columns (they are not in the feature set).

**`engineer_features(df)`**
Add five derived columns:
- `CREDIT_INCOME_RATIO` = AMT_CREDIT / (AMT_INCOME_TOTAL + 1)
- `ANNUITY_INCOME_RATIO` = AMT_ANNUITY / (AMT_INCOME_TOTAL + 1)
- `AGE_YEARS` = -DAYS_BIRTH / 365
- `YEARS_EMPLOYED` = -DAYS_EMPLOYED / 365
- `CREDIT_TERM` = AMT_CREDIT / (AMT_ANNUITY + 1)

**`FEATURE_COLS` constant**
A module-level list of exactly 25 column names used for training. Includes the original numeric columns from application, the 5 engineered features, and the 5 bureau aggregates. This list is the single source of truth for column selection — both training and app inference must use it.

**`split_data(df, target_col, test_size=0.2, random_state=42)`**
Stratified train/test split. Returns `(X_train, X_test, y_train, y_test)`.

**`load_dataset(app_path, bureau_path)`**
Full pipeline: load both CSVs → aggregate bureau → left join on `SK_ID_CURR` → fill bureau null columns with 0 → `clean_application` → `encode_categoricals` → `engineer_features` → select `FEATURE_COLS` → save `FEATURE_COLS` to `outputs/feature_cols.json` → `split_data`. Returns `(X_train, X_test, y_train, y_test)`.

### Tests to write in `tests/test_data_prep.py`

Use small synthetic DataFrames (5 rows) that mimic the schema of each real table. Do not load real CSVs in tests.

- `test_load_bureau_aggregates_shape` — verify output has one row per unique `SK_ID_CURR` and all 5 aggregate columns present.
- `test_clean_application_handles_days_employed_anomaly` — verify no row has `DAYS_EMPLOYED == 365243` after cleaning.
- `test_clean_application_imputes_nulls` — verify `AMT_INCOME_TOTAL`, `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3` have zero nulls after cleaning.
- `test_encode_categoricals_no_string_columns` — verify no object-dtype columns remain after encoding.
- `test_engineer_features_adds_ratios` — verify `CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`, `AGE_YEARS`, `YEARS_EMPLOYED`, `CREDIT_TERM` are all present.
- `test_split_data_stratified` — with a 200-row synthetic DataFrame (8% positive), verify split sizes are 160/40 and both splits contain positive examples.

### Steps

- [ ] **Step 1: Write all 6 failing tests** in `tests/test_data_prep.py`.
- [ ] **Step 2: Run tests** — verify all 6 FAIL with `ModuleNotFoundError`.
- [ ] **Step 3: Implement `data_prep.py`** with all functions described above.
- [ ] **Step 4: Run tests** — verify all 6 PASS.
- [ ] **Step 5: Smoke test against real data** — run `load_dataset()` and print `X_train.shape` (expect ~240K rows, 25 columns) and positive rate (expect ~0.08).
- [ ] **Step 6: Commit** with message `feat: join application + bureau, clean, encode, engineer features`.

---

## Task 3: Model Training — Three Classifiers

**Files:**
- Create: `src/train.py`
- Create: `tests/test_train.py`

### What `train.py` must do

**`train_xgboost(X_train, y_train)`**
XGBClassifier with: `n_estimators=200`, `max_depth=5`, `learning_rate=0.05`, `scale_pos_weight=neg/pos` (computed from `y_train`), `eval_metric='logloss'`, `random_state=42`, `n_jobs=-1`.

**`train_random_forest(X_train, y_train)`**
RandomForestClassifier with: `n_estimators=100`, `class_weight='balanced'`, `random_state=42`, `n_jobs=-1`.

**`train_logistic_regression(X_train, y_train)`**
LogisticRegression with: `C=0.1`, `solver='lbfgs'`, `class_weight='balanced'`, `max_iter=1000`, `random_state=42`.

**`evaluate_model(model, X_test, y_test)`**
Returns a dict with keys: `roc_auc`, `f1`, `precision`, `recall`, `confusion_matrix`.

**`save_model(model, path)` / `load_model(path)`**
Serialize/deserialize using joblib. Create parent directory if missing.

**`run_training_pipeline(app_path, bureau_path)`**
Calls `load_dataset`, trains all three models, saves each to `models/`, evaluates each, saves metrics to `outputs/metrics.csv`, prints the metrics table. Callable as `python src/train.py`.

### Tests to write in `tests/test_train.py`

Use a small synthetic dataset (500 rows, 10 features, ~8% positive rate). Do not load real CSVs in tests.

- `test_train_xgboost_returns_model` — verify `model.predict(X_test)` returns an array of length `len(X_test)`.
- `test_train_random_forest_returns_model` — same check.
- `test_train_logistic_regression_returns_model` — same check.
- `test_evaluate_model_returns_metrics` — verify all four metric keys present and `roc_auc` is between 0 and 1.

### Steps

- [ ] **Step 1: Write all 4 failing tests** in `tests/test_train.py`.
- [ ] **Step 2: Run tests** — verify all 4 FAIL with `ModuleNotFoundError`.
- [ ] **Step 3: Implement `train.py`** with all functions described above.
- [ ] **Step 4: Run tests** — verify all 4 PASS.
- [ ] **Step 5: Run `python src/train.py`** against real data. Verify `models/*.pkl` and `outputs/metrics.csv` are created. XGBoost ROC-AUC should be ≥ 0.75.
- [ ] **Step 6: Commit** with message `feat: train XGBoost, RandomForest, LogisticRegression; save models and metrics`.

---

## Task 4: SHAP Explainability Layer

**Files:**
- Create: `src/explain.py`
- Create: `tests/test_explain.py`

### What `explain.py` must do

**`compute_shap_values(model, X)`**
Create a `shap.TreeExplainer` from the XGBoost model. Call `.shap_values(X)`. Return `(shap_values_array, explainer)`. The returned array must have the same shape as `X`.

**`plot_shap_summary(shap_values, X, save_path, max_display=15)`**
Beeswarm summary plot showing top 15 features by mean absolute SHAP value across all rows in `X`. Save to `save_path` as PNG at 150 dpi. Use `matplotlib.use('Agg')` to avoid display errors in headless environments.

**`plot_shap_waterfall(explainer, X, idx=0, save_path)`**
Waterfall plot for the single row at `X.iloc[idx]`. Save to `save_path` as PNG.

**`run_explain_pipeline(model_path, app_path, bureau_path, sample_size=1000)`**
Load model from `model_path`, load data via `load_dataset`, sample 1000 rows from `X_test`, compute SHAP values, save both plots to `outputs/`. Callable as `python src/explain.py`.

### Tests to write in `tests/test_explain.py`

Use a small synthetic dataset (50 rows, 10 features). Train a quick XGBoost model inline within each test — do not load real models or data.

- `test_compute_shap_values_shape` — verify returned array shape equals `X.shape`.
- `test_plot_shap_summary_saves_file` — verify PNG file exists at `tmp_path` after calling `plot_shap_summary`.
- `test_plot_shap_waterfall_saves_file` — verify PNG file exists at `tmp_path` after calling `plot_shap_waterfall`.

### Steps

- [ ] **Step 1: Write all 3 failing tests** in `tests/test_explain.py`.
- [ ] **Step 2: Run tests** — verify all 3 FAIL with `ModuleNotFoundError`.
- [ ] **Step 3: Implement `explain.py`** with all functions described above.
- [ ] **Step 4: Run tests** — verify all 3 PASS.
- [ ] **Step 5: Run `python src/explain.py`** against real data. Verify both PNGs are created. `EXT_SOURCE_2` and `EXT_SOURCE_3` should appear near the top of the summary plot.
- [ ] **Step 6: Commit** with message `feat: SHAP global summary and local waterfall plots`.

---

## Task 5: Streamlit UI

**Files:**
- Create: `app.py`

### What `app.py` must do

**Model and artifact loading** (cached with `st.cache_resource` / `st.cache_data`):
- Load all 3 models from `models/`
- Load `FEATURE_COLS` from `outputs/feature_cols.json`
- Load metrics from `outputs/metrics.csv`

**Sidebar inputs** (the applicant profile):
- Contract type (selectbox: Cash loans / Revolving loans)
- Gender (selectbox: M / F)
- Owns car, owns realty (selectboxes)
- Number of children (slider 0–10)
- Annual income, loan amount, annual repayment, goods price (number inputs)
- Age in years (slider 18–70)
- Years employed (slider 0–40)
- External Score 2 and External Score 3 (sliders 0.0–1.0) — most predictive features
- Bureau record count and max days overdue (sliders)

**Input assembly:**
Build a raw input DataFrame matching the schema expected by `encode_categoricals` and `engineer_features`. Run both functions on it. Align columns to `FEATURE_COLS` (filling any missing with 0). This ensures identical preprocessing to training.

**Main panel — left column:**
- `st.metric` showing risk score (0–100), computed as `int((1 - default_probability) * 100)`
- Risk tier badge: Low Risk (score ≥ 70, green), Medium Risk (40–69, orange), High Risk (< 40, red)
- `st.progress` bar as a visual gauge
- Caption showing raw default probability as a percentage

**Main panel — right column:**
- SHAP waterfall plot for the current applicant's input, rendered via `st.pyplot`

**Below the columns:**
- Divider, then model comparison table from `outputs/metrics.csv` with 4 decimal formatting
- Global SHAP summary PNG loaded from `outputs/shap_summary.png` via `st.image`

### Steps

- [ ] **Step 1: Implement `app.py`** with all sections described above.
- [ ] **Step 2: Run `streamlit run app.py`** and manually verify: sidebar sliders update the risk score and waterfall in real time; model comparison table appears; global SHAP summary image loads.
- [ ] **Step 3: Commit** with message `feat: Streamlit UI with risk score, SHAP waterfall, and model comparison`.

---

## Task 6: README

**Files:**
- Create: `README.md`

### What README.md must contain

- Project title and one-sentence description
- Results table with ROC-AUC, F1, Precision, Recall for all 3 models (fill from `outputs/metrics.csv`)
- ASCII architecture diagram showing the data flow: two CSVs → data_prep → train/explain → models/outputs → app
- Brief description of each source file's responsibility
- Setup section: `pip install`, dataset download note, `python src/train.py`, `python src/explain.py`, `streamlit run app.py`
- Live demo link (placeholder until deployed)
- Dataset credit with link to Kaggle competition
- Note on future work: remaining 5 tables not yet joined

### Steps

- [ ] **Step 1: Write `README.md`** with all sections above.
- [ ] **Step 2: Fill in the results table** with real values from `outputs/metrics.csv`.
- [ ] **Step 3: Commit** with message `docs: README with architecture, setup, results table`.

---

## Task 7: Deploy to Streamlit Community Cloud

- [ ] **Step 1: Push repo to GitHub** — create a new public repo named `home-credit-risk-scorer` and push `main`.
- [ ] **Step 2: Commit model files and outputs** — if any model file exceeds 100 MB, set up Git LFS (`git lfs track "models/*.pkl"`) before committing. Otherwise commit directly.
- [ ] **Step 3: Deploy on Streamlit Community Cloud** — go to share.streamlit.io, create new app, point to the repo and `app.py`, click Deploy.
- [ ] **Step 4: Update README** with the live URL and push.

---

## Task 8 (Optional): GenAI Insight Layer

**Files:**
- Create: `src/llm_summary.py`

> Only implement after Tasks 1–7 are complete and tested.

### What `llm_summary.py` must do

**`build_risk_prompt(shap_values, feature_names, prob)`**
Extract the top 3 features by absolute SHAP value. Format into a one-sentence structured prompt that includes the risk level (low/medium/high), the default probability, and each factor with a direction (increases/decreases risk).

**`generate_risk_summary(prompt, model_name='distilgpt2')`**
Use `transformers.pipeline('text-generation')` with `distilgpt2` to generate a sentence continuation. Return only the generated text after the prompt, stripped of whitespace.

### Steps

- [ ] **Step 1: Add `transformers` and `torch` to `requirements.txt`** and install them.
- [ ] **Step 2: Implement `llm_summary.py`** with both functions above.
- [ ] **Step 3: Wire into `app.py`** — add an "AI Risk Summary" section below the waterfall plot that calls `build_risk_prompt` with the current applicant's SHAP values and `generate_risk_summary` inside a `st.spinner`.
- [ ] **Step 4: Run `streamlit run app.py`** and verify the AI Risk Summary section appears with a readable sentence.
- [ ] **Step 5: Commit** with message `feat: GenAI NL risk summary using HuggingFace distilgpt2`.

---

## Self-Review Checklist

- [x] Two-table join (application + bureau) covered in Task 2
- [x] `DAYS_EMPLOYED == 365243` anomaly handled in `clean_application`
- [x] `FEATURE_COLS` defined once in `data_prep.py`, saved to `outputs/feature_cols.json`, loaded by `app.py` — no column mismatch between training and inference
- [x] `app.py` runs the same `encode_categoricals` + `engineer_features` pipeline as training — no train/serve skew
- [x] All 3 classifiers trained and compared
- [x] SHAP global + local plots saved as PNGs and shown in Streamlit
- [x] No code in this plan — descriptions only
- [x] Function names consistent across all tasks: `load_dataset`, `train_xgboost`, `compute_shap_values`, `FEATURE_COLS`
