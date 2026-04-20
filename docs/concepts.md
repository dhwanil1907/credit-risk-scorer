# Concepts Used in This Project

A reference guide covering every technique, algorithm, and tool applied in the Home Credit Risk Scorer.

---

## 1. The Problem: Binary Credit Default Prediction

**Credit default prediction** is a supervised classification task. Given a borrower's financial history and loan application details, we predict whether they will fail to repay (default) within the loan period.

- **Target variable:** `TARGET` — 1 (client with payment difficulties), 0 (all other)
- **Class imbalance:** Only ~8% of applicants default. A naive model that always predicts "no default" gets 92% accuracy but is useless. We must account for this.
- **Evaluation metric:** ROC-AUC is preferred over accuracy because it measures the model's ability to rank risky applicants above safe ones, independent of any threshold. Top Kaggle scores on this competition are ~0.80.

---

## 2. The Dataset: Home Credit Default Risk

**Source:** Kaggle competition. We use two of the seven available tables.

### application_train.csv (~300K rows, 122 columns)
The main table. One row per loan application.

Key features used:

| Feature | Description |
|---|---|
| `AMT_INCOME_TOTAL` | Applicant's annual income |
| `AMT_CREDIT` | Loan amount requested |
| `AMT_ANNUITY` | Annual loan repayment amount |
| `AMT_GOODS_PRICE` | Price of goods the loan is for |
| `DAYS_BIRTH` | Negative integer: days before application date the person was born (e.g. -10000 = ~27 years old) |
| `DAYS_EMPLOYED` | Negative integer: days before application the person started current job; **365243 is an anomalous placeholder for unemployed/pensioners** |
| `EXT_SOURCE_1/2/3` | Normalized scores from external data sources (credit bureaus, etc.) — most predictive features in the dataset |
| `CODE_GENDER` | M / F / XNA |
| `NAME_CONTRACT_TYPE` | Cash loans vs. Revolving loans |
| `FLAG_OWN_CAR` | Whether applicant owns a car (Y/N) |
| `REGION_RATING_CLIENT` | Rating of the region where client lives (1/2/3) |

### bureau.csv (~1.7M rows, 17 columns)
Each row is one historical credit record from the credit bureau. An applicant can have many bureau records. We aggregate this into one row per applicant before joining.

Key aggregations produced:

| Aggregate | Description |
|---|---|
| `bureau_count` | Total number of previous credits |
| `bureau_active_count` | Number of currently active credits |
| `bureau_max_overdue` | Maximum days past due on any bureau credit |
| `bureau_total_debt` | Sum of outstanding debt across all bureau credits |
| `bureau_avg_credit` | Average credit amount across bureau records |

### Why only 2 of 7 tables?
The other 5 tables (POS_CASH_balance, credit_card_balance, installments_payments, previous_application, bureau_balance) add more signal but at significantly higher data engineering cost. For a portfolio project scoped to ~300 lines of code per file, `application_train + bureau` gives us the richer-than-single-CSV story without drowning in joins.

---

## 3. Multi-Table Join

The bureau table has a many-to-one relationship with the application table: one applicant (`SK_ID_CURR`) can appear in bureau many times (one row per past credit). Before joining, we must **aggregate** bureau down to one row per `SK_ID_CURR` using `groupby`.

```
bureau.csv (1.7M rows)
    groupby SK_ID_CURR → agg → bureau_agg (300K rows, one per applicant)
        ↓ left join on SK_ID_CURR
application_train.csv (300K rows)
    → merged_df (300K rows, richer feature set)
```

A **left join** is used so applicants with no bureau records are kept (their bureau columns are filled with 0, meaning "no bureau history").

---

## 4. Data Preprocessing

### DAYS_EMPLOYED Anomaly
`DAYS_EMPLOYED = 365243` is a special placeholder value used for pensioners and unemployed applicants — it is not a real employment duration. It must be replaced with `NaN` before imputation. If left in, it would wildly skew the median and mislead the model.

### DAYS_BIRTH → AGE_YEARS
`DAYS_BIRTH` is stored as a negative number (days *before* the application date). To get age in years: `AGE_YEARS = -DAYS_BIRTH / 365`. This makes the feature interpretable.

### Null Imputation with Median
All remaining null numeric values are filled with the column median. Median is preferred over mean for financial data because income and credit amounts are right-skewed — a few very large values would drag the mean upward and produce a poor imputation.

### Categorical Encoding
Four categorical columns are label-encoded with explicit mappings:
- `CODE_GENDER`: M=1, F=0, XNA=0
- `NAME_CONTRACT_TYPE`: Cash loans=1, Revolving loans=0
- `FLAG_OWN_CAR`: Y=1, N=0
- `FLAG_OWN_REALTY`: Y=1, N=0

All other object columns are dropped (they would require one-hot encoding and add many columns; out of scope for this version).

### Stratified Train/Test Split (80/20)
A regular random split could put all (or none) of the rare positive cases in the test set. **Stratified splitting** ensures the 8% default rate is preserved in both splits. We use `train_test_split(..., stratify=y)` from scikit-learn.

---

## 5. Feature Engineering

### CREDIT_INCOME_RATIO
```
AMT_CREDIT / (AMT_INCOME_TOTAL + 1)
```
How many years of income does this loan represent? A loan of $400K on an income of $50K/year is far riskier than the same loan on $200K/year.

### ANNUITY_INCOME_RATIO
```
AMT_ANNUITY / (AMT_INCOME_TOTAL + 1)
```
What fraction of annual income goes to loan repayments? High values indicate the borrower is stretched thin.

### YEARS_EMPLOYED
```
-DAYS_EMPLOYED / 365
```
Converts negative days to positive years. Employment stability is a strong predictor of repayment ability.

### CREDIT_TERM
```
AMT_CREDIT / (AMT_ANNUITY + 1)
```
Implicit loan term in years. Longer terms mean smaller payments but more total interest — different risk profiles.

---

## 6. Train/Serve Consistency (No Feature Leakage)

A common mistake in ML systems is transforming data differently at training time vs. inference time. We prevent this by:

1. Defining `FEATURE_COLS` as a single constant in `data_prep.py`
2. Saving `FEATURE_COLS` to `outputs/feature_cols.json` after training
3. Loading `feature_cols.json` in `app.py` and passing `app_input[FEATURE_COLS]` to the model — same columns, same order, every time

The app runs the **same** `encode_categoricals` and `engineer_features` functions on user inputs before calling `predict_proba`. This ensures training and serving are always aligned.

---

## 7. Machine Learning Models

### 7.1 Logistic Regression
Models the log-odds of default as a linear combination of features:
```
log(p / 1-p) = β₀ + β₁x₁ + β₂x₂ + ...
```
**Strengths:** Fast, interpretable coefficients, well-calibrated probabilities, useful as a baseline.
**Weaknesses:** Cannot capture non-linear relationships (e.g., the interaction between income and loan amount).
**Key hyperparameter:** `C=0.1` — inverse regularization strength. Low C → strong L2 penalty → simpler, less overfit model.

### 7.2 Random Forest
An ensemble of decision trees trained on random subsets of data and features (bootstrap aggregation). Each tree votes; majority wins.

**Strengths:** Handles non-linearity and feature interactions naturally, robust to outliers.
**Weaknesses:** Slower than XGBoost on large datasets, less accurate on structured tabular data.
**Key hyperparameter:** `class_weight='balanced'` — reweights each sample inversely proportional to class frequency, addressing imbalance without modifying the data.

**How bagging works:** Each tree is trained on a bootstrap sample (random sample with replacement). At each split, only a random subset of features is considered. This de-correlates the trees so their errors cancel out in aggregation.

### 7.3 XGBoost (Extreme Gradient Boosting)
The strongest model in this project. Uses **boosting** — trees are trained sequentially, each correcting the errors of the previous ensemble.

**How gradient boosting works:**
1. Start with a constant prediction (log-odds of the base rate, ~0.08 for this dataset).
2. Compute pseudo-residuals (gradient of the loss w.r.t. current predictions).
3. Fit a shallow tree to predict those residuals.
4. Add the tree to the ensemble, scaled by learning rate.
5. Repeat for `n_estimators` rounds.

**Key hyperparameters used:**
- `n_estimators=200` — 200 boosting rounds
- `max_depth=5` — shallow trees to prevent overfitting
- `learning_rate=0.05` — shrinks each tree's contribution; trades speed for accuracy
- `scale_pos_weight = neg/pos` — tells XGBoost to penalize missing a default more heavily; equivalent to weighting positive examples by this factor in the loss function

**Why XGBoost dominates tabular data:**
- Built-in L1/L2 regularization on tree weights
- Handles missing values natively (learns which branch to take for NaN)
- Second-order gradient optimization (uses curvature information, not just slope)
- Parallel tree construction

---

## 8. Evaluation Metrics

### ROC-AUC
The **ROC curve** plots True Positive Rate vs. False Positive Rate at every possible classification threshold. AUC = 1.0 is perfect; AUC = 0.5 is random guessing.

**Interpretation:** AUC = 0.76 means "there is a 76% chance that a randomly chosen defaulter is scored riskier than a randomly chosen non-defaulter." This is why ROC-AUC is the primary metric: it measures ranking quality across all possible decision thresholds.

### F1 Score
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
Useful when both false positives (rejecting a good applicant) and false negatives (approving a defaulter) have real costs.

### Precision
Of all applicants predicted to default, what fraction actually did? High precision = few false alarms (fewer good applicants incorrectly rejected).

### Recall (Sensitivity)
Of all applicants who actually defaulted, what fraction did we catch? High recall = few missed defaults.

### The Precision-Recall Trade-off
Lowering the classification threshold increases recall (catch more defaulters) but decreases precision (more false alarms). The optimal threshold depends on the business cost of each type of error — out of scope for this project but important in production.

---

## 9. SHAP (SHapley Additive exPlanations)

SHAP answers: **"Why did the model give this applicant a high risk score?"**

### Shapley Values (Game Theory Origin)
From cooperative game theory: features are "players" cooperating to produce a prediction. The Shapley value of a feature is its **average marginal contribution** to the prediction across all possible orderings of features.

Formally:
```
ϕᵢ = Σ (|S|!(n-|S|-1)!/n!) × [f(S ∪ {i}) - f(S)]
```
where the sum is over all subsets S of features not containing feature i, and f(S) is the model prediction using only the features in S.

### TreeExplainer
`shap.TreeExplainer` computes exact Shapley values for tree-based models (XGBoost, Random Forest) by exploiting the tree structure. It runs in O(TLD²) time where T=trees, L=leaves, D=depth — much faster than the exponential brute-force.

For each prediction:
```
prediction = base_value + SHAP(EXT_SOURCE_2) + SHAP(CREDIT_INCOME_RATIO) + ... + SHAP(bureau_max_overdue)
```
where `base_value` = mean prediction across the training set (~0.08 for this dataset).

### Global Explanation: Beeswarm Summary Plot
Shows the distribution of SHAP values for every feature across a sample of applicants:
- Each dot = one applicant
- X-axis = SHAP value (positive = pushes toward default, negative = pushes toward safe)
- Color = feature value (red = high, blue = low)
- Features sorted by mean absolute SHAP value (most important at top)

Expected result: `EXT_SOURCE_2`, `EXT_SOURCE_3`, and `CREDIT_INCOME_RATIO` should appear near the top.

### Local Explanation: Waterfall Plot
For a **single applicant**, shows which features pushed the prediction above (red bars) or below (blue bars) the baseline. Answers: "why is this specific person flagged as high risk?"

### Why SHAP Matters in Fintech
Regulators require lenders to provide "adverse action notices" — written reasons why someone was denied credit. SHAP values provide a mathematically grounded, defensible basis for those reasons. This is the difference between a model that predicts and a model that explains.

---

## 10. Class Imbalance Handling

Three complementary techniques:

| Technique | Where Used | Mechanism |
|---|---|---|
| `scale_pos_weight = neg/pos` | XGBoost | Multiplies the gradient update for each positive-class sample by this factor (~11.5×) |
| `class_weight='balanced'` | RF, LR | Scikit-learn reweights each sample's loss contribution inversely to class frequency |
| Stratified split | All | Ensures the 8% default rate is preserved in both train and test sets |

**Why not SMOTE (synthetic oversampling)?**
SMOTE generates synthetic minority-class samples by interpolating between existing ones. Tree-based models with weighted loss functions handle imbalance more cleanly and don't require modifying the training set distribution.

---

## 11. Model Persistence with joblib

`joblib.dump(model, path)` serializes a trained model object to a binary file. `joblib.load(path)` restores it. joblib is preferred over the standard library alternative for ML objects because it efficiently handles large NumPy arrays (the weight matrices in Random Forest and XGBoost can be hundreds of MB).

**Security note:** Only load model files from sources you trust — deserializing binary ML artifacts from untrusted origins can be unsafe, as with any binary serialization format.

---

## 12. Streamlit

Streamlit turns a Python script into an interactive web app. No HTML or JavaScript required.

**Key concepts used:**

- **`st.cache_resource`:** Caches expensive Python objects (models, feature column lists) across reruns. Without this, the XGBoost model would reload from disk on every slider interaction.
- **`st.cache_data`:** Caches data objects (DataFrames) across reruns.
- **Reactivity model:** Every widget change triggers a full top-to-bottom re-execution of `app.py`. Code order matters — inputs must be defined before they are used.
- **`st.sidebar`:** Side panel for input controls, keeping the main panel clean.
- **`st.metric`:** Displays a large number with a label — used for the risk score.
- **`st.progress`:** Visual bar — used as a score gauge (0–100).

---

## 13. GenAI Insight Layer (Optional — Phase 6)

### HuggingFace Transformers
HuggingFace provides thousands of pre-trained language models through a unified `pipeline` API. We use `text-generation` with `distilgpt2` — a distilled (smaller, faster) version of GPT-2 that runs locally with no API key.

### SHAP → Prompt → Natural Language
The pipeline:
1. Extract the top 3 SHAP features for the applicant by absolute value.
2. Format into a structured prompt: *"Summarize in one sentence: applicant has high credit default risk (68% probability). Key factors: EXT_SOURCE_2 (decreases risk), CREDIT_INCOME_RATIO (increases risk), bureau_max_overdue (increases risk)."*
3. Feed to distilgpt2 to generate a sentence continuation.
4. Display below the waterfall plot in Streamlit.

**Why distilgpt2 and not a larger model?**
This is a portfolio project — local, free, and offline-capable is the right constraint. The prompt is highly structured so even a small model produces readable output. In a production system you would swap in a hosted model with a system prompt instructing it to write in plain language suitable for applicants.

---

## 14. Key ML Engineering Principles Applied

### Single Source of Truth for Feature Columns
`FEATURE_COLS` is defined once in `data_prep.py`, saved to `outputs/feature_cols.json` after training, and loaded by `app.py` at runtime. There is no duplicated column list to get out of sync.

### Train/Test Discipline
Models are evaluated only on the held-out test set. Hyperparameters are not tuned against test performance — that would be **data leakage**.

### Reproducibility
All stochastic operations (`train_test_split`, `XGBClassifier`, `RandomForestClassifier`, `LogisticRegression`) accept `random_state=42` so results are identical across runs and machines.

### Separation of Concerns
- `data_prep.py` owns data transformation — nothing else touches raw CSVs
- `train.py` owns model artifacts — nothing else saves `.pkl` files
- `explain.py` owns SHAP computation — nothing else instantiates `TreeExplainer`
- `app.py` is a consumer only — loads artifacts, never creates them
