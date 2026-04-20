import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Single source of truth for all feature columns used in training and inference.
# Both train.py and app.py must use this list — ensures no column mismatch.
FEATURE_COLS = [
    # --- Raw numeric columns from application_train.csv ---
    "AMT_CREDIT",               # Loan amount
    "AMT_INCOME_TOTAL",         # Applicant's annual income
    "AMT_ANNUITY",              # Annual loan repayment amount
    "AMT_GOODS_PRICE",          # Price of goods the loan is for
    "DAYS_BIRTH",               # Age in days (negative — days before application)
    "DAYS_EMPLOYED",            # Employment duration in days (negative = currently employed)
    "EXT_SOURCE_1",             # External credit score 1 (0–1, higher = better)
    "EXT_SOURCE_2",             # External credit score 2 — top predictor
    "EXT_SOURCE_3",             # External credit score 3 — top predictor
    "CNT_CHILDREN",             # Number of children
    "REGION_POPULATION_RELATIVE",  # Population density of client's region
    "DAYS_ID_PUBLISH",          # Days since ID was last changed
    "OWN_CAR_AGE",              # Age of the applicant's car in years
    "CNT_FAM_MEMBERS",          # Family size
    "REGION_RATING_CLIENT",     # Bureau rating of the region (1–3)
    "REGION_RATING_CLIENT_W_CITY",  # Same but city-weighted

    # --- Encoded categorical columns (label-encoded to integers) ---
    "CODE_GENDER",              # M→1, F→0, XNA→0
    "NAME_CONTRACT_TYPE",       # Cash loans→1, Revolving loans→0
    "FLAG_OWN_CAR",             # Y→1, N→0
    "FLAG_OWN_REALTY",          # Y→1, N→0

    # --- Engineered features (derived from raw columns) ---
    "CREDIT_INCOME_RATIO",      # Loan amount relative to income
    "ANNUITY_INCOME_RATIO",     # Monthly repayment burden relative to income
    "AGE_YEARS",                # Age in years (positive, converted from DAYS_BIRTH)
    "YEARS_EMPLOYED",           # Years at current job (positive)
    "CREDIT_TERM",              # Implied loan term in months

    # --- Bureau aggregates (joined from bureau.csv) ---
    "bureau_count",                 # Total number of past credit records
    "bureau_active_count",          # Number of currently active credits
    "bureau_closed_count",          # Closed bureau lines (stability signal)
    "bureau_max_overdue",           # Worst overdue days across all bureau records
    "bureau_mean_overdue",          # Average overdue severity
    "bureau_total_debt",            # Total outstanding debt across bureau records
    "bureau_avg_credit",            # Average credit line size
    "bureau_max_line_credit",       # Largest single bureau line
    "bureau_sum_limit",             # Total credit limit across lines
    "bureau_sum_overdue_amt",       # Total overdue amount (severity)
    "bureau_mean_days_credit",      # Mean DAYS_CREDIT (recency / history length)
    "bureau_credit_type_nunique",   # Diversity of credit products
    "bureau_mean_line_annuity",     # Mean bureau-line annuity
    "bureau_max_max_overdue",       # Worst historical max-overdue cap
    "bureau_cnt_prolong_sum",       # Total prolongations (stress signal)

    # --- Optional: previous_application.csv ---
    "prev_app_count",               # Number of prior applications
    "prev_amt_credit_sum",          # Sum of prior offered credits
    "prev_amt_credit_mean",         # Mean prior offered credit
    "prev_amt_application_sum",     # Sum of prior application amounts
    "prev_annuity_sum",             # Sum of prior annuities
    "prev_down_payment_sum",        # Sum of down payments on prior apps

    # --- Optional: installments_payments.csv ---
    "inst_count",                   # Total number of instalment records
    "inst_days_late_mean",          # Mean days payment was late (DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT)
    "inst_days_late_max",           # Worst single late payment in days
    "inst_pct_late",                # Fraction of payments made late
    "inst_underpay_mean",           # Mean underpayment amount (AMT_INSTALMENT - AMT_PAYMENT, clipped at 0)
    "inst_underpay_sum",            # Total amount underpaid across all instalments

    # --- Optional: credit_card_balance.csv ---
    "cc_count",                     # Number of monthly balance snapshots
    "cc_utilization_mean",          # Mean (balance / credit limit) — key stress signal
    "cc_utilization_max",           # Peak utilization
    "cc_dpd_mean",                  # Mean days past due
    "cc_dpd_max",                   # Worst days past due
    "cc_drawings_mean",             # Mean total drawings per month
]


def load_bureau_aggregates(bureau_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse bureau.csv from one-row-per-credit to one-row-per-applicant.

    bureau.csv has multiple rows per SK_ID_CURR (one per credit line).
    Aggregations are guarded by column presence so tests and thin CSVs still work.
    """
    b = bureau_df
    cols = set(bureau_df.columns)
    count_key = "SK_ID_BUREAU" if "SK_ID_BUREAU" in cols else "SK_ID_CURR"
    spec: dict = {"bureau_count": (count_key, "count")}

    if "CREDIT_ACTIVE" in cols:
        spec["bureau_active_count"] = ("CREDIT_ACTIVE", lambda s: (s.astype(str) == "Active").sum())
        spec["bureau_closed_count"] = ("CREDIT_ACTIVE", lambda s: (s.astype(str) == "Closed").sum())

    if "CREDIT_DAY_OVERDUE" in cols:
        spec["bureau_max_overdue"] = ("CREDIT_DAY_OVERDUE", "max")
        spec["bureau_mean_overdue"] = ("CREDIT_DAY_OVERDUE", "mean")

    if "AMT_CREDIT_SUM_DEBT" in cols:
        spec["bureau_total_debt"] = ("AMT_CREDIT_SUM_DEBT", "sum")

    if "AMT_CREDIT_SUM" in cols:
        spec["bureau_avg_credit"] = ("AMT_CREDIT_SUM", "mean")
        spec["bureau_max_line_credit"] = ("AMT_CREDIT_SUM", "max")

    if "AMT_CREDIT_SUM_LIMIT" in cols:
        spec["bureau_sum_limit"] = ("AMT_CREDIT_SUM_LIMIT", "sum")

    if "AMT_CREDIT_SUM_OVERDUE" in cols:
        spec["bureau_sum_overdue_amt"] = ("AMT_CREDIT_SUM_OVERDUE", "sum")

    if "DAYS_CREDIT" in cols:
        spec["bureau_mean_days_credit"] = ("DAYS_CREDIT", "mean")

    if "CREDIT_TYPE" in cols:
        spec["bureau_credit_type_nunique"] = ("CREDIT_TYPE", "nunique")

    if "AMT_ANNUITY" in cols:
        spec["bureau_mean_line_annuity"] = ("AMT_ANNUITY", "mean")

    if "AMT_CREDIT_MAX_OVERDUE" in cols:
        spec["bureau_max_max_overdue"] = ("AMT_CREDIT_MAX_OVERDUE", "max")

    if "CNT_CREDIT_PROLONG" in cols:
        spec["bureau_cnt_prolong_sum"] = ("CNT_CREDIT_PROLONG", "sum")

    return b.groupby("SK_ID_CURR").agg(**spec).reset_index()


def load_previous_application_aggregates(prev_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per SK_ID_CURR from previous_application.csv (prior loan applications).

    File is optional — place next to application_train.csv after downloading from Kaggle
    (Home Credit Default Risk): previous_application.csv.
    """
    cols = set(prev_df.columns)
    # One count per prior row; prefer SK_ID_PREV when present (one id per prior application).
    count_col = "SK_ID_PREV" if "SK_ID_PREV" in cols else "SK_ID_CURR"
    spec: dict[str, tuple] = {"prev_app_count": (count_col, "count")}

    if "AMT_CREDIT" in cols:
        spec["prev_amt_credit_sum"] = ("AMT_CREDIT", "sum")
        spec["prev_amt_credit_mean"] = ("AMT_CREDIT", "mean")

    if "AMT_APPLICATION" in cols:
        spec["prev_amt_application_sum"] = ("AMT_APPLICATION", "sum")

    if "AMT_ANNUITY" in cols:
        spec["prev_annuity_sum"] = ("AMT_ANNUITY", "sum")

    if "AMT_DOWN_PAYMENT" in cols:
        spec["prev_down_payment_sum"] = ("AMT_DOWN_PAYMENT", "sum")

    return prev_df.groupby("SK_ID_CURR").agg(**spec).reset_index()


def load_installments_aggregates(inst_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate installments_payments.csv to one row per applicant.

    Key signals:
    - days_late: positive = paid late, negative = paid early. Mean/max lateness captures
      chronic vs occasional late payers.
    - underpay: how much short of the required instalment was each payment? Chronic
      underpayment is a strong default signal even when payments aren't technically missed.
    """
    df = inst_df.copy()

    # Positive = paid late, negative = paid early
    df["days_late"] = df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]

    # Underpayment: how much less than required was paid (floor at 0 — overpayments don't count)
    df["underpay"] = (df["AMT_INSTALMENT"] - df["AMT_PAYMENT"]).clip(lower=0)

    agg = df.groupby("SK_ID_CURR").agg(
        inst_count=("SK_ID_PREV", "count"),
        inst_days_late_mean=("days_late", "mean"),
        inst_days_late_max=("days_late", "max"),
        inst_pct_late=("days_late", lambda x: (x > 0).mean()),   # fraction of late payments
        inst_underpay_mean=("underpay", "mean"),
        inst_underpay_sum=("underpay", "sum"),
    ).reset_index()
    return agg


def load_credit_card_aggregates(cc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate credit_card_balance.csv to one row per applicant.

    Key signals:
    - utilization: balance / credit_limit. High utilization = credit-stressed.
      Clipped to [0, 1] to handle data entry errors where balance > limit.
    - SK_DPD: days past due per statement. Any DPD > 0 is a delinquency signal.
    """
    df = cc_df.copy()

    # Utilization rate: balance as fraction of credit limit (capped at 1 to remove outliers)
    df["utilization"] = (
        df["AMT_BALANCE"] / (df["AMT_CREDIT_LIMIT_ACTUAL"] + 1)
    ).clip(0, 1)

    agg = df.groupby("SK_ID_CURR").agg(
        cc_count=("SK_ID_PREV", "count"),
        cc_utilization_mean=("utilization", "mean"),
        cc_utilization_max=("utilization", "max"),
        cc_dpd_mean=("SK_DPD", "mean"),
        cc_dpd_max=("SK_DPD", "max"),
        cc_drawings_mean=("AMT_DRAWINGS_CURRENT", "mean"),
    ).reset_index()
    return agg


def clean_application(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix known data quality issues in application_train.csv.

    - DAYS_EMPLOYED == 365243 is a placeholder for unemployed/pensioners, not a real value.
      Replace it with NaN so it doesn't corrupt ratios and model features.
    - Impute all remaining numeric nulls with the column median (robust to outliers).
    """
    df = df.copy()

    # 365243 is a sentinel value Kaggle uses for "not employed" — treat as missing
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    # Median imputation for all numeric columns that have missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert string categorical columns to integers using explicit mappings.

    Only the four columns listed below are part of FEATURE_COLS. All other
    object-dtype columns (e.g. occupation, housing type) are dropped since
    they aren't in the feature set and would break numeric model inputs.
    """
    df = df.copy()

    # Explicit label mappings — kept deterministic (no fit/transform needed at inference)
    gender_map = {"M": 1, "F": 0, "XNA": 0}
    contract_map = {"Cash loans": 1, "Revolving loans": 0}
    car_map = {"Y": 1, "N": 0}
    realty_map = {"Y": 1, "N": 0}

    if "CODE_GENDER" in df.columns:
        df["CODE_GENDER"] = df["CODE_GENDER"].map(gender_map)
    if "NAME_CONTRACT_TYPE" in df.columns:
        df["NAME_CONTRACT_TYPE"] = df["NAME_CONTRACT_TYPE"].map(contract_map)
    if "FLAG_OWN_CAR" in df.columns:
        df["FLAG_OWN_CAR"] = df["FLAG_OWN_CAR"].map(car_map)
    if "FLAG_OWN_REALTY" in df.columns:
        df["FLAG_OWN_REALTY"] = df["FLAG_OWN_REALTY"].map(realty_map)

    # Drop any columns still holding string values (not in our feature set)
    object_cols = df.select_dtypes(include="object").columns
    df = df.drop(columns=object_cols)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features that capture economic ratios and convert day-based
    columns into human-readable units.

    The +1 in denominators prevents division by zero for edge cases (zero income, zero annuity).
    """
    df = df.copy()

    # How large is the loan relative to income? High ratio = more financial strain
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)

    # What fraction of income goes to repayment each year?
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)

    # DAYS_BIRTH is negative (days before application), flip sign and convert to years
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365

    # DAYS_EMPLOYED is negative for currently employed; flip sign → years at job
    df["YEARS_EMPLOYED"] = -df["DAYS_EMPLOYED"] / 365

    # Implied repayment term in months (loan size / monthly payment)
    df["CREDIT_TERM"] = df["AMT_CREDIT"] / (df["AMT_ANNUITY"] + 1)

    return df


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Stratified 80/20 train/test split.

    Stratify=y ensures the ~8% default rate is preserved in both splits,
    preventing an accidental imbalance that would skew evaluation metrics.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def load_dataset(app_path: str, bureau_path: str):
    """
    Full data pipeline from raw CSVs to train/test splits ready for modeling.

    Optional extra tables are merged when present in the same folder as application_train.csv:
    - previous_application.csv — prior loan application history
    - installments_payments.csv — payment punctuality and underpayment behaviour
    - credit_card_balance.csv  — credit utilization and delinquency signals

    Steps:
        1. Load application + bureau CSVs
        2. Aggregate bureau (and optional previous_application) to one row per applicant
        3. Left-join onto application (missing history → zeros for numeric aggregates)
        4. Clean anomalies and impute nulls
        5. Encode categoricals
        6. Engineer new features
        7. Select FEATURE_COLS and save the list to outputs/feature_cols.json
        8. Stratified split → return (X_train, X_test, y_train, y_test)
    """
    app_df = pd.read_csv(app_path)
    bureau_df = pd.read_csv(bureau_path)
    data_dir = Path(app_path).parent

    # Aggregate bureau then join — many applicants won't appear in bureau at all
    bureau_agg = load_bureau_aggregates(bureau_df)
    df = app_df.merge(bureau_agg, on="SK_ID_CURR", how="left")

    bureau_join_cols = [c for c in bureau_agg.columns if c != "SK_ID_CURR"]
    df[bureau_join_cols] = df[bureau_join_cols].fillna(0)

    prev_path = data_dir / "previous_application.csv"
    if prev_path.is_file():
        prev_agg = load_previous_application_aggregates(pd.read_csv(prev_path))
        df = df.merge(prev_agg, on="SK_ID_CURR", how="left")
        prev_join_cols = [c for c in prev_agg.columns if c != "SK_ID_CURR"]
        df[prev_join_cols] = df[prev_join_cols].fillna(0)

    # Applicants with no instalment history get zeros (no payment behaviour on record)
    inst_path = data_dir / "installments_payments.csv"
    if inst_path.is_file():
        inst_agg = load_installments_aggregates(pd.read_csv(inst_path))
        df = df.merge(inst_agg, on="SK_ID_CURR", how="left")
        inst_join_cols = [c for c in inst_agg.columns if c != "SK_ID_CURR"]
        df[inst_join_cols] = df[inst_join_cols].fillna(0)

    # Applicants with no credit card history get zeros
    cc_path = data_dir / "credit_card_balance.csv"
    if cc_path.is_file():
        cc_agg = load_credit_card_aggregates(pd.read_csv(cc_path))
        df = df.merge(cc_agg, on="SK_ID_CURR", how="left")
        cc_join_cols = [c for c in cc_agg.columns if c != "SK_ID_CURR"]
        df[cc_join_cols] = df[cc_join_cols].fillna(0)

    df = clean_application(df)
    df = encode_categoricals(df)
    df = engineer_features(df)

    target = df["TARGET"]

    # Guard against FEATURE_COLS listing a column not present in this dataset version
    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_cols]

    # Persist the exact column list so app.py can align inference inputs at runtime
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/feature_cols.json", "w") as f:
        json.dump(available_cols, f)

    return split_data(X, target)
