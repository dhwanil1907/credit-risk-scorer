import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────────────────────────────────────
# FEATURE_COLS: the definitive list of inputs the model uses to make predictions
#
# Think of this as the "form" the model fills in for every applicant.
# Every column here is a number the model has learned to weigh.
# Both the training step (train.py) and the web app (app.py) must use this
# exact same list in the exact same order — otherwise the model will be
# reading the wrong numbers for the wrong fields.
# ──────────────────────────────────────────────────────────────────────────────
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
    # The model can only work with numbers, so text categories are converted:
    "CODE_GENDER",              # M→1, F→0, XNA→0
    "NAME_CONTRACT_TYPE",       # Cash loans→1, Revolving loans→0
    "FLAG_OWN_CAR",             # Y→1, N→0
    "FLAG_OWN_REALTY",          # Y→1, N→0

    # --- Engineered features (derived from raw columns) ---
    # These are calculated columns that capture relationships the model finds useful
    "CREDIT_INCOME_RATIO",      # Loan amount relative to income
    "ANNUITY_INCOME_RATIO",     # Monthly repayment burden relative to income
    "AGE_YEARS",                # Age in years (positive, converted from DAYS_BIRTH)
    "YEARS_EMPLOYED",           # Years at current job (positive)
    "CREDIT_TERM",              # Implied loan term in months
    "EXT_MEAN",                     # Composite external credit score
    "EXT_MEAN_X_ANNUITY_RATIO",     # Interaction: credit score × repayment burden
    "EXT_MEAN_X_CREDIT_RATIO",      # Interaction: credit score × loan-to-income ratio
    "DEBT_STRESS",                  # Combined overextension: annuity ratio + credit ratio

    # --- Bureau aggregates (joined from bureau.csv) ---
    # Each applicant may have many rows in bureau.csv (one per loan/credit product).
    # These columns collapse that history into a single summary per applicant.
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
    # Summarises all loan applications this person has made before (accepted or not)
    "prev_app_count",               # Number of prior applications
    "prev_amt_credit_sum",          # Sum of prior offered credits
    "prev_amt_credit_mean",         # Mean prior offered credit
    "prev_amt_application_sum",     # Sum of prior application amounts
    "prev_annuity_sum",             # Sum of prior annuities
    "prev_down_payment_sum",        # Sum of down payments on prior apps

    # --- Optional: installments_payments.csv ---
    # Summarises whether the applicant paid their previous loans on time
    "inst_count",                   # Total number of instalment records
    "inst_days_late_mean",          # Mean days payment was late (DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT)
    "inst_days_late_max",           # Worst single late payment in days
    "inst_pct_late",                # Fraction of payments made late
    "inst_underpay_mean",           # Mean underpayment amount (AMT_INSTALMENT - AMT_PAYMENT, clipped at 0)
    "inst_underpay_sum",            # Total amount underpaid across all instalments

    # --- Optional: credit_card_balance.csv ---
    # Summarises the applicant's credit card usage and payment behaviour
    "cc_count",                     # Number of monthly balance snapshots
    "cc_utilization_mean",          # Mean (balance / credit limit) — key stress signal
    "cc_utilization_max",           # Peak utilization
    "cc_dpd_mean",                  # Mean days past due
    "cc_dpd_max",                   # Worst days past due
    "cc_drawings_mean",             # Mean total drawings per month
]


def load_bureau_aggregates(bureau_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise each applicant's credit bureau history into a single row.

    The raw bureau data has one row for every credit product a person has
    ever had (e.g. a car loan, a credit card, a mortgage — each gets its
    own row). This function collapses all of those rows so we end up with
    one neat summary row per person, ready to be joined to their main
    application.

    Aggregations are guarded by column presence so tests and thin CSVs still work.
    """
    b = bureau_df
    cols = set(bureau_df.columns)

    # Decide which column to count — SK_ID_BUREAU is the unique ID per credit line
    count_key = "SK_ID_BUREAU" if "SK_ID_BUREAU" in cols else "SK_ID_CURR"

    # Start building the summary: how many bureau records does this person have?
    spec: dict = {"bureau_count": (count_key, "count")}

    # How many of their credits are still active vs fully paid off?
    if "CREDIT_ACTIVE" in cols:
        spec["bureau_active_count"] = ("CREDIT_ACTIVE", lambda s: (s.astype(str) == "Active").sum())
        spec["bureau_closed_count"] = ("CREDIT_ACTIVE", lambda s: (s.astype(str) == "Closed").sum())

    # Overdue days: has the person been behind on payments? By how much on average?
    if "CREDIT_DAY_OVERDUE" in cols:
        spec["bureau_max_overdue"] = ("CREDIT_DAY_OVERDUE", "max")
        spec["bureau_mean_overdue"] = ("CREDIT_DAY_OVERDUE", "mean")

    # Total outstanding debt across all credit lines
    if "AMT_CREDIT_SUM_DEBT" in cols:
        spec["bureau_total_debt"] = ("AMT_CREDIT_SUM_DEBT", "sum")

    # Size of credit lines — how large are the loans they typically take?
    if "AMT_CREDIT_SUM" in cols:
        spec["bureau_avg_credit"] = ("AMT_CREDIT_SUM", "mean")
        spec["bureau_max_line_credit"] = ("AMT_CREDIT_SUM", "max")

    # Total credit limit the person has been granted across all products
    if "AMT_CREDIT_SUM_LIMIT" in cols:
        spec["bureau_sum_limit"] = ("AMT_CREDIT_SUM_LIMIT", "sum")

    # Total money currently overdue across all credit lines
    if "AMT_CREDIT_SUM_OVERDUE" in cols:
        spec["bureau_sum_overdue_amt"] = ("AMT_CREDIT_SUM_OVERDUE", "sum")

    # How long has the person had credit? (signals experience/stability)
    if "DAYS_CREDIT" in cols:
        spec["bureau_mean_days_credit"] = ("DAYS_CREDIT", "mean")

    # How diverse is their credit history? (mortgages, cards, car loans etc.)
    if "CREDIT_TYPE" in cols:
        spec["bureau_credit_type_nunique"] = ("CREDIT_TYPE", "nunique")

    # Average annual repayment amount across their bureau credits
    if "AMT_ANNUITY" in cols:
        spec["bureau_mean_line_annuity"] = ("AMT_ANNUITY", "mean")

    # Worst-ever overdue amount on a single credit line
    if "AMT_CREDIT_MAX_OVERDUE" in cols:
        spec["bureau_max_max_overdue"] = ("AMT_CREDIT_MAX_OVERDUE", "max")

    # How many times did they ask to extend/delay repayment? More = financial stress
    if "CNT_CREDIT_PROLONG" in cols:
        spec["bureau_cnt_prolong_sum"] = ("CNT_CREDIT_PROLONG", "sum")

    # Group by applicant ID and compute all the summaries above in one step
    return b.groupby("SK_ID_CURR").agg(**spec).reset_index()


def load_previous_application_aggregates(prev_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise each applicant's history of previous loan applications into one row.

    People often apply for multiple loans over their lifetime. This file has
    one row per prior application. We collapse them into a single summary per
    person — for example: how many times have they applied? How much credit
    have they been offered in total?

    This file is optional — place it next to application_train.csv after
    downloading from Kaggle (Home Credit Default Risk): previous_application.csv.
    """
    cols = set(prev_df.columns)

    # SK_ID_PREV is a unique ID per prior application (one person can have many)
    count_col = "SK_ID_PREV" if "SK_ID_PREV" in cols else "SK_ID_CURR"

    # Start with: how many previous applications has this person made?
    spec: dict[str, tuple] = {"prev_app_count": (count_col, "count")}

    # Total and average credit amounts offered across all previous applications
    if "AMT_CREDIT" in cols:
        spec["prev_amt_credit_sum"] = ("AMT_CREDIT", "sum")
        spec["prev_amt_credit_mean"] = ("AMT_CREDIT", "mean")

    # Total of what the person asked for (vs what was offered)
    if "AMT_APPLICATION" in cols:
        spec["prev_amt_application_sum"] = ("AMT_APPLICATION", "sum")

    # Total of all annual repayment amounts across previous loans
    if "AMT_ANNUITY" in cols:
        spec["prev_annuity_sum"] = ("AMT_ANNUITY", "sum")

    # Total of all down payments made on previous applications
    if "AMT_DOWN_PAYMENT" in cols:
        spec["prev_down_payment_sum"] = ("AMT_DOWN_PAYMENT", "sum")

    # Group by applicant and produce one summary row per person
    return prev_df.groupby("SK_ID_CURR").agg(**spec).reset_index()


def load_installments_aggregates(inst_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise each applicant's loan repayment behaviour into one row.

    This file has one row for every scheduled payment on every prior loan.
    For example, if someone had a 12-month loan and a 24-month loan, this
    file would have 36 rows for them.

    We calculate two key signals:
    - Lateness: did they pay on time, or were they consistently late?
      A positive number means they paid after the due date (late).
      A negative number means they paid before the due date (early).
    - Underpayment: did they pay the full required amount?
      If someone pays less than required each month, that is a warning sign
      even if they don't technically miss a payment.
    """
    df = inst_df.copy()

    # Positive = paid late, negative = paid early
    # This is calculated by comparing the actual payment date to the due date
    df["days_late"] = df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]

    # Underpayment: how much less than required was paid each time?
    # We floor this at zero — overpaying is fine and shouldn't count negatively
    df["underpay"] = (df["AMT_INSTALMENT"] - df["AMT_PAYMENT"]).clip(lower=0)

    # Collapse all payment rows into one summary per applicant
    agg = df.groupby("SK_ID_CURR").agg(
        inst_count=("SK_ID_PREV", "count"),           # Total number of payment records
        inst_days_late_mean=("days_late", "mean"),     # Average lateness across all payments
        inst_days_late_max=("days_late", "max"),       # Worst single late payment
        inst_pct_late=("days_late", lambda x: (x > 0).mean()),   # What fraction of payments were late?
        inst_underpay_mean=("underpay", "mean"),       # Average shortfall per payment
        inst_underpay_sum=("underpay", "sum"),         # Total money never paid across all instalments
    ).reset_index()
    return agg


def load_credit_card_aggregates(cc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise each applicant's credit card usage behaviour into one row.

    This file has one row per monthly credit card statement per applicant.
    We calculate two key signals:
    - Utilization: what fraction of their credit limit are they using?
      High utilization (e.g. 90% of limit) signals financial stress.
      We cap this at 100% to handle any data errors where balance exceeds limit.
    - Days past due (DPD): how many days overdue was the balance each month?
      Any DPD > 0 means a missed or late payment on the credit card.
    """
    df = cc_df.copy()

    # Utilization rate: how full is their credit card? (capped at 1.0 = 100%)
    # We add 1 to the limit to avoid dividing by zero if the limit is missing
    df["utilization"] = (
        df["AMT_BALANCE"] / (df["AMT_CREDIT_LIMIT_ACTUAL"] + 1)
    ).clip(0, 1)

    # Collapse all monthly statements into one summary per applicant
    agg = df.groupby("SK_ID_CURR").agg(
        cc_count=("SK_ID_PREV", "count"),              # How many monthly snapshots exist?
        cc_utilization_mean=("utilization", "mean"),   # Average monthly credit card usage rate
        cc_utilization_max=("utilization", "max"),     # Highest usage rate ever recorded
        cc_dpd_mean=("SK_DPD", "mean"),                # Average days overdue per month
        cc_dpd_max=("SK_DPD", "max"),                  # Worst overdue month ever
        cc_drawings_mean=("AMT_DRAWINGS_CURRENT", "mean"),  # Average monthly spending on the card
    ).reset_index()
    return agg


def clean_application(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix known data quality issues in the main application file.

    Two problems are addressed:
    1. The value 365243 in the "days employed" column is a fake placeholder
       that the data provider uses to indicate the person is not currently
       employed (e.g. a pensioner or homemaker). It is not a real number of
       days. We replace it with a blank (missing value) so it does not corrupt
       any calculations.
    2. Some other numeric columns have missing values. We fill those blanks
       with the middle value (median) of that column — a safe, common approach
       that is not thrown off by extreme outliers.
    """
    df = df.copy()

    # 365243 is a sentinel/placeholder value meaning "not employed" — treat as missing
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    # Fill all remaining missing numbers with the median of their column
    # (median = the middle value when sorted, robust to extreme high/low outliers)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert text-based columns into numbers the model can understand.

    Machine learning models only work with numbers, not words. For example,
    "Male" and "Female" must become 1 and 0. We use fixed, predetermined
    mappings (not learned from data) so that the encoding is always the same
    at training time and at prediction time — consistency is essential.

    Only the four columns below are converted because they are in our feature
    list. All other text columns (e.g. occupation type, housing type) are
    removed because they are not used by this model.
    """
    df = df.copy()

    # Fixed mappings from text labels to numbers
    gender_map = {"M": 1, "F": 0, "XNA": 0}           # Male=1, Female=0, Unknown=0
    contract_map = {"Cash loans": 1, "Revolving loans": 0}  # Cash loan=1, Revolving=0
    car_map = {"Y": 1, "N": 0}                          # Owns car: Yes=1, No=0
    realty_map = {"Y": 1, "N": 0}                       # Owns property: Yes=1, No=0

    # Apply each mapping only if the column exists (handles partial datasets)
    if "CODE_GENDER" in df.columns:
        df["CODE_GENDER"] = df["CODE_GENDER"].map(gender_map)
    if "NAME_CONTRACT_TYPE" in df.columns:
        df["NAME_CONTRACT_TYPE"] = df["NAME_CONTRACT_TYPE"].map(contract_map)
    if "FLAG_OWN_CAR" in df.columns:
        df["FLAG_OWN_CAR"] = df["FLAG_OWN_CAR"].map(car_map)
    if "FLAG_OWN_REALTY" in df.columns:
        df["FLAG_OWN_REALTY"] = df["FLAG_OWN_REALTY"].map(realty_map)

    # Remove any remaining text columns — they are not part of the model's feature set
    object_cols = df.select_dtypes(include="object").columns
    df = df.drop(columns=object_cols)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new calculated columns that capture financial relationships.

    Raw numbers alone can be misleading. For example, a loan of $250,000
    is very different for someone earning $30,000/year vs $300,000/year.
    These new columns express the data as ratios and in more human-readable
    units (years instead of days), which helps the model learn better.

    Note: we add +1 to denominators throughout to avoid dividing by zero
    in edge cases where income or annuity is recorded as zero.
    """
    df = df.copy()

    # How large is the loan compared to the person's annual income?
    # A ratio of 5 means the loan is 5× their annual income — high financial strain
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)

    # What fraction of annual income goes to loan repayments each year?
    # A ratio of 0.3 means 30% of income is consumed by repayments
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)

    # Convert age from days to years and make it a positive number
    # (The raw data stores age as a negative number of days before the application date)
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365

    # Convert employment duration from days to years and make it positive
    # (The raw data stores this as a negative number for currently employed people)
    df["YEARS_EMPLOYED"] = -df["DAYS_EMPLOYED"] / 365

    # Implied loan duration in months: if you borrow $120,000 and repay $10,000/year,
    # the implied term is 12 months
    df["CREDIT_TERM"] = df["AMT_CREDIT"] / (df["AMT_ANNUITY"] + 1)

    # Composite external credit score (average of all three sources)
    ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in df.columns]
    df["EXT_MEAN"] = df[ext_cols].mean(axis=1) if ext_cols else np.nan

    # Interaction: captures repayment burden for high-scoring applicants.
    # A high value means good credit score but also heavy repayment obligations.
    df["EXT_MEAN_X_ANNUITY_RATIO"] = df["EXT_MEAN"] * df["ANNUITY_INCOME_RATIO"]

    # Interaction: captures loan-to-income strain for high-scoring applicants.
    # A high value means good credit score but also extreme loan relative to income.
    df["EXT_MEAN_X_CREDIT_RATIO"] = df["EXT_MEAN"] * df["CREDIT_INCOME_RATIO"]

    # Combined debt stress signal: total financial overextension
    df["DEBT_STRESS"] = df["ANNUITY_INCOME_RATIO"] + df["CREDIT_INCOME_RATIO"]

    return df


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Divide the dataset into a training set (80%) and a testing set (20%).

    The model learns from the training set and is evaluated on the testing set.
    The testing set is never used during training — it simulates real-world
    applicants the model hasn't seen before.

    We use stratified splitting, which means the proportion of defaulters
    (~8%) is preserved in both halves. Without this, one half might
    accidentally contain far fewer defaulters, making evaluation misleading.
    """
    # stratify=y ensures the default rate (~8%) is the same in both train and test
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def load_dataset(app_path: str, bureau_path: str):
    """
    Run the complete data preparation pipeline from raw files to model-ready data.

    This is the main entry point for data loading. It reads the raw CSV files,
    joins supporting tables, fixes data issues, converts text to numbers,
    creates calculated columns, and splits into training and testing sets.

    Optional supplementary files are used if they exist in the same folder:
    - previous_application.csv — prior loan applications this person made
    - installments_payments.csv — whether past loan payments were made on time
    - credit_card_balance.csv  — credit card usage and payment history

    Steps performed in order:
        1. Load the main application file and the credit bureau file
        2. Summarise bureau history to one row per applicant and join it on
        3. Join any optional files (previous applications, payment history, credit cards)
        4. Fix data quality issues and fill missing values
        5. Convert text columns to numbers
        6. Add calculated ratio and unit-conversion columns
        7. Select only the columns the model will use, and save that list to disk
        8. Split into training (80%) and testing (20%) sets and return them
    """
    app_df = pd.read_csv(app_path)
    bureau_df = pd.read_csv(bureau_path)
    data_dir = Path(app_path).parent

    # Summarise bureau history and attach it to the main application table.
    # People who have no bureau records get zeros (no prior credit history on file).
    bureau_agg = load_bureau_aggregates(bureau_df)
    df = app_df.merge(bureau_agg, on="SK_ID_CURR", how="left")

    # Fill any gaps from the bureau join with zero (person has no bureau history)
    bureau_join_cols = [c for c in bureau_agg.columns if c != "SK_ID_CURR"]
    df[bureau_join_cols] = df[bureau_join_cols].fillna(0)

    # Join previous applications if the file exists in the data folder
    prev_path = data_dir / "previous_application.csv"
    if prev_path.is_file():
        prev_agg = load_previous_application_aggregates(pd.read_csv(prev_path))
        df = df.merge(prev_agg, on="SK_ID_CURR", how="left")
        prev_join_cols = [c for c in prev_agg.columns if c != "SK_ID_CURR"]
        # People with no prior applications get zeros
        df[prev_join_cols] = df[prev_join_cols].fillna(0)

    # Join instalment payment history if the file exists
    # Applicants with no instalment history get zeros (no payment behaviour on record)
    inst_path = data_dir / "installments_payments.csv"
    if inst_path.is_file():
        inst_agg = load_installments_aggregates(pd.read_csv(inst_path))
        df = df.merge(inst_agg, on="SK_ID_CURR", how="left")
        inst_join_cols = [c for c in inst_agg.columns if c != "SK_ID_CURR"]
        df[inst_join_cols] = df[inst_join_cols].fillna(0)

    # Join credit card history if the file exists
    # Applicants with no credit card history get zeros
    cc_path = data_dir / "credit_card_balance.csv"
    if cc_path.is_file():
        cc_agg = load_credit_card_aggregates(pd.read_csv(cc_path))
        df = df.merge(cc_agg, on="SK_ID_CURR", how="left")
        cc_join_cols = [c for c in cc_agg.columns if c != "SK_ID_CURR"]
        df[cc_join_cols] = df[cc_join_cols].fillna(0)

    # Fix data quality issues (e.g. the fake 365243 employment value) and fill missing numbers
    df = clean_application(df)

    # Convert text categories (e.g. "Male"/"Female") to numbers the model can use
    df = encode_categoricals(df)

    # Add calculated columns like "loan as % of income"
    df = engineer_features(df)

    # TARGET is the column we are trying to predict: 1 = defaulted, 0 = repaid
    target = df["TARGET"]

    # Use only the columns in our agreed feature list.
    # Some optional files may not be present, so we only include what is available.
    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_cols]

    # Save the exact list of columns used so the web app can load the same list
    # and guarantee it sends the model data in the correct column order
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/feature_cols.json", "w") as f:
        json.dump(available_cols, f)

    # Return training and testing sets (features and labels separately)
    return split_data(X, target)
