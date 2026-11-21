import json
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
    # Only features the dashboard/API form actually collects are used, so the model sees the
    # same kind of inputs at training and serving time. Gender is deliberately excluded.

    # --- Raw numeric columns from application_train.csv ---
    "AMT_CREDIT",               # Loan amount
    "AMT_INCOME_TOTAL",         # Applicant's annual income
    "AMT_ANNUITY",              # Loan repayment amount per period
    "AMT_GOODS_PRICE",          # Price of goods the loan is for
    "DAYS_BIRTH",               # Age in days (negative — days before application)
    "DAYS_EMPLOYED",            # Employment duration in days (negative = currently employed)
    "EXT_SOURCE_1",             # External credit score 1 (0–1, higher = better)
    "EXT_SOURCE_2",             # External credit score 2 — top predictor
    "EXT_SOURCE_3",             # External credit score 3 — top predictor
    "CNT_CHILDREN",             # Number of children

    # --- Encoded categorical columns (label-encoded to integers) ---
    "NAME_CONTRACT_TYPE",       # Cash loans→1, Revolving loans→0
    "FLAG_OWN_CAR",             # Y→1, N→0
    "FLAG_OWN_REALTY",          # Y→1, N→0

    # --- Engineered features (derived from raw columns) ---
    "CREDIT_INCOME_RATIO",      # Loan amount relative to income
    "ANNUITY_INCOME_RATIO",     # Repayment burden relative to income
    "AGE_YEARS",                # Age in years (positive, converted from DAYS_BIRTH)
    "YEARS_EMPLOYED",           # Years at current job (positive)
    "CREDIT_TERM",              # Implied loan term in periods
    "EXT_MEAN",                     # Composite external credit score
    "EXT_MEAN_X_ANNUITY_RATIO",     # Interaction: credit score × repayment burden
    "EXT_MEAN_X_CREDIT_RATIO",      # Interaction: credit score × loan-to-income ratio
    "DEBT_STRESS",                  # Combined overextension: annuity ratio + credit ratio

    # --- Bureau aggregates (joined from bureau.csv) ---
    "bureau_count",                 # Total number of past credit records
    "bureau_max_overdue",           # Worst overdue days across all bureau records
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

    Only application_train.csv and bureau.csv are read: the model uses just the
    features the dashboard/API form can supply (see FEATURE_COLS).

    Steps performed in order:
        1. Load the main application file and the credit bureau file
        2. Summarise bureau history to one row per applicant and join it on
        3. Fix data quality issues and fill missing values
        4. Convert text columns to numbers
        5. Add calculated ratio and unit-conversion columns
        6. Select only the columns the model will use, and save that list to disk
        7. Split into training (80%) and testing (20%) sets and return them
    """
    app_df = pd.read_csv(app_path)
    bureau_df = pd.read_csv(bureau_path)

    # Summarise bureau history and attach it to the main application table.
    # People who have no bureau records get zeros (no prior credit history on file).
    bureau_agg = load_bureau_aggregates(bureau_df)
    df = app_df.merge(bureau_agg, on="SK_ID_CURR", how="left")

    # Fill any gaps from the bureau join with zero (person has no bureau history)
    bureau_join_cols = [c for c in bureau_agg.columns if c != "SK_ID_CURR"]
    df[bureau_join_cols] = df[bureau_join_cols].fillna(0)

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
    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "feature_cols.json", "w") as f:
        json.dump(available_cols, f)

    # Return training and testing sets (features and labels separately)
    return split_data(X, target)
