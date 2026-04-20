# =============================================================================
# Tests for src/data_prep.py
#
# These tests check that the data preparation steps work correctly BEFORE
# touching any real data. They use small, hand-crafted fake datasets (5 rows)
# so tests run in milliseconds and don't depend on Kaggle files being present.
#
# How to run: python -m pytest tests/test_data_prep.py -v
# =============================================================================

import numpy as np
import pandas as pd
import pytest

from src.data_prep import (
    clean_application,
    encode_categoricals,
    engineer_features,
    load_bureau_aggregates,
    load_previous_application_aggregates,
    split_data,
)


# ---------------------------------------------------------------------------
# Fake data builders
# These create small sample tables that look like the real Kaggle CSVs
# but with just 5 rows so tests stay fast and predictable.
# ---------------------------------------------------------------------------

def make_bureau_df():
    """
    Fake credit bureau table: 5 rows across 3 applicants (SK_ID_CURR 1, 2, 3).
    Applicant 1 has 2 credit lines, applicant 3 has 2 credit lines, applicant 2 has 1.
    This lets us verify that the aggregation correctly groups by applicant.
    """
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 2, 3, 3],           # applicant ID (links to application table)
            "SK_ID_BUREAU": [101, 102, 201, 301, 302], # unique ID for each credit line
            "CREDIT_ACTIVE": ["Active", "Closed", "Active", "Active", "Active"],  # is the credit line open?
            "CREDIT_DAY_OVERDUE": [0, 5, 10, 0, 3],  # how many days past due on this line
            "AMT_CREDIT_SUM_DEBT": [1000.0, 200.0, 500.0, 800.0, 100.0],  # outstanding debt
            "AMT_CREDIT_SUM": [5000.0, 3000.0, 2000.0, 4000.0, 1500.0],   # total credit limit
            "AMT_CREDIT_SUM_LIMIT": [6000.0, 3500.0, 2500.0, 4500.0, 2000.0],  # available limit
            "AMT_CREDIT_SUM_OVERDUE": [0.0, 10.0, 0.0, 0.0, 5.0],         # overdue amount in currency
            "DAYS_CREDIT": [-100, -200, -300, -400, -500],  # when credit was opened (negative = days ago)
            "CREDIT_TYPE": ["Consumer credit"] * 5,          # type of credit product
            "AMT_ANNUITY": [100.0, 200.0, 150.0, 120.0, 80.0],  # monthly repayment on this line
            "AMT_CREDIT_MAX_OVERDUE": [0.0, 5.0, 0.0, 0.0, 2.0],  # worst overdue amount ever
            "CNT_CREDIT_PROLONG": [0, 1, 0, 0, 0],  # how many times repayment was extended (stress signal)
        }
    )


def make_application_df():
    """
    Fake application table: 5 applicants with deliberately messy data.

    Intentional issues included so we can test cleaning:
    - Row 2 and 4: DAYS_EMPLOYED = 365243 (fake value meaning 'not employed')
    - Row 2: AMT_INCOME_TOTAL is None (missing)
    - Row 3: AMT_ANNUITY is None (missing)
    - Rows 2, 3, 4: EXT_SOURCE scores have missing values
    """
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3, 4, 5],
            "TARGET": [0, 1, 0, 0, 1],           # 0 = repaid loan, 1 = defaulted
            "DAYS_EMPLOYED": [-500, 365243, -200, 365243, -1000],  # negative = employed; 365243 = not employed
            "DAYS_BIRTH": [-10000, -12000, -15000, -8000, -20000],  # age in days (negative = days before today)
            "AMT_CREDIT": [100000.0, 200000.0, 150000.0, 80000.0, 300000.0],   # loan amount
            "AMT_INCOME_TOTAL": [50000.0, None, 75000.0, 30000.0, 100000.0],   # annual income (one missing)
            "AMT_ANNUITY": [5000.0, 10000.0, None, 4000.0, 15000.0],           # annual repayment (one missing)
            "EXT_SOURCE_1": [0.5, None, 0.3, 0.7, 0.4],   # external credit score (some missing)
            "EXT_SOURCE_2": [0.6, 0.4, None, 0.8, 0.3],
            "EXT_SOURCE_3": [0.7, 0.5, 0.2, None, 0.6],
            "CODE_GENDER": ["M", "F", "XNA", "M", "F"],   # gender as text — will be converted to numbers
            "NAME_CONTRACT_TYPE": [
                "Cash loans",
                "Revolving loans",
                "Cash loans",
                "Cash loans",
                "Revolving loans",
            ],
            "FLAG_OWN_CAR": ["Y", "N", "Y", "N", "Y"],       # owns a car (Y/N → 1/0)
            "FLAG_OWN_REALTY": ["Y", "N", "N", "Y", "Y"],    # owns property (Y/N → 1/0)
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_load_bureau_aggregates_shape():
    """
    Check that bureau aggregation produces exactly one row per applicant
    and all expected summary columns are present.

    We have 3 unique applicants in the fake data, so the result must have 3 rows.
    We also verify all 15 aggregate columns were created (counts, overdue amounts, etc.)
    """
    bureau_df = make_bureau_df()
    result = load_bureau_aggregates(bureau_df)

    # One row per applicant — no duplicates, no missing applicants
    assert result.shape[0] == 3, "Should have one row per unique SK_ID_CURR"

    # All the summary columns the model expects must be present
    expected_cols = {
        "bureau_count",             # total number of credit lines
        "bureau_active_count",      # how many are still open
        "bureau_closed_count",      # how many are closed
        "bureau_max_overdue",       # worst overdue days
        "bureau_mean_overdue",      # average overdue days
        "bureau_total_debt",        # total money owed
        "bureau_avg_credit",        # average credit line size
        "bureau_max_line_credit",   # largest single credit line
        "bureau_sum_limit",         # total available credit
        "bureau_sum_overdue_amt",   # total overdue amount in currency
        "bureau_mean_days_credit",  # average age of credit lines
        "bureau_credit_type_nunique",  # number of different credit product types
        "bureau_mean_line_annuity", # average monthly repayment across lines
        "bureau_max_max_overdue",   # worst single overdue amount ever
        "bureau_cnt_prolong_sum",   # total number of repayment extensions
    }
    assert expected_cols.issubset(result.columns), "Missing aggregate columns"


def test_load_previous_application_aggregates_shape() -> None:
    """
    Check that previous application aggregation works correctly.

    Applicant 1 made 2 prior loan applications (rows 0 and 1).
    Applicant 2 made 1 prior application (row 2).
    We verify:
    - the output has the right columns
    - applicant 1's application count equals 2
    """
    # Fake previous applications table: 2 applications for applicant 1, 1 for applicant 2
    prev = pd.DataFrame(
        {
            "SK_ID_PREV": [10, 11, 12],           # unique ID per prior application
            "SK_ID_CURR": [1, 1, 2],              # links back to the applicant
            "AMT_CREDIT": [1000.0, 2000.0, 3000.0],         # how much credit was offered
            "AMT_APPLICATION": [900.0, 1900.0, 2800.0],     # how much they applied for
            "AMT_ANNUITY": [100.0, 200.0, 300.0],           # annual repayment on prior loan
            "AMT_DOWN_PAYMENT": [50.0, 0.0, 100.0],         # down payment made
        }
    )
    out = load_previous_application_aggregates(prev)

    # The output must include applicant ID and at least the count and credit sum columns
    assert {"SK_ID_CURR", "prev_app_count", "prev_amt_credit_sum"}.issubset(out.columns)

    # Applicant 1 had 2 prior applications — the count must reflect that
    assert out.loc[out["SK_ID_CURR"] == 1, "prev_app_count"].iloc[0] == 2


def test_clean_application_handles_days_employed_anomaly():
    """
    Verify that the fake employment value 365243 is replaced during cleaning.

    365243 appears in the dataset as a placeholder for unemployed/pensioners —
    it's not a real number of days. If left in, it would distort every ratio
    that uses employment duration. After cleaning, no row should have this value.
    """
    df = make_application_df().copy()
    cleaned = clean_application(df)

    # No row should still have 365243 — it must be gone (replaced with NaN then imputed)
    assert not (cleaned["DAYS_EMPLOYED"] == 365243).any(), (
        "365243 anomaly should be replaced with NaN"
    )


def test_clean_application_imputes_nulls():
    """
    Verify that missing values (None/NaN) are filled in after cleaning.

    The fake data has intentional gaps in income and external credit scores.
    After cleaning, every column should be complete — no blanks.
    The missing values are filled with the median of each column.
    """
    df = make_application_df().copy()
    cleaned = clean_application(df)

    # These columns had missing values in our fake data — all must be filled after cleaning
    for col in ["AMT_INCOME_TOTAL", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
        assert cleaned[col].isna().sum() == 0, f"{col} should have no nulls after cleaning"


def test_encode_categoricals_no_string_columns():
    """
    Verify that all text columns are converted to numbers after encoding.

    Machine learning models can't handle text directly — they need numbers.
    After encode_categoricals(), columns like 'M'/'F' become 1/0,
    and any other text columns that aren't needed are dropped entirely.
    """
    df = make_application_df().copy()
    df = clean_application(df)
    encoded = encode_categoricals(df)

    # No column should still contain text values
    object_cols = encoded.select_dtypes(include="object").columns.tolist()
    assert len(object_cols) == 0, f"Object columns remain after encoding: {object_cols}"


def test_engineer_features_adds_ratios():
    """
    Verify that all 5 derived (calculated) columns are created.

    These new columns are more informative than the raw values:
    - CREDIT_INCOME_RATIO: loan size as a multiple of income
    - ANNUITY_INCOME_RATIO: annual repayment as a fraction of income
    - AGE_YEARS: age in years (easier to interpret than days)
    - YEARS_EMPLOYED: job tenure in years
    - CREDIT_TERM: implied number of months to repay
    """
    df = make_application_df().copy()
    df = clean_application(df)
    df = encode_categoricals(df)
    engineered = engineer_features(df)

    expected = {
        "CREDIT_INCOME_RATIO",
        "ANNUITY_INCOME_RATIO",
        "AGE_YEARS",
        "YEARS_EMPLOYED",
        "CREDIT_TERM",
    }
    assert expected.issubset(engineered.columns), f"Missing engineered features: {expected - set(engineered.columns)}"


def test_split_data_stratified():
    """
    Verify that the 80/20 train/test split works correctly and keeps defaults in both halves.

    We create a fake dataset of 200 rows where exactly 16 are defaults (8%).
    After splitting:
    - 160 rows should go to training (80%)
    - 40 rows should go to testing (20%)
    - Both splits must contain at least one default — 'stratified' splitting
      ensures the 8% default rate is preserved in both halves, not accidentally
      concentrated in one.
    """
    rng = np.random.default_rng(42)
    n = 200
    y = np.zeros(n, dtype=int)                   # start with all zeros (no defaults)
    pos_idx = rng.choice(n, size=16, replace=False)
    y[pos_idx] = 1                               # set 16 random rows to 1 (default)

    # 5 random numeric columns — content doesn't matter for this test
    X = pd.DataFrame(rng.random((n, 5)), columns=[f"f{i}" for i in range(5)])
    y_series = pd.Series(y)

    X_train, X_test, y_train, y_test = split_data(X, y_series)

    # Size checks
    assert len(X_train) == 160, f"Expected 160 train rows, got {len(X_train)}"
    assert len(X_test) == 40, f"Expected 40 test rows, got {len(X_test)}"

    # Both splits must have at least one default — stratified split guarantees this
    assert y_train.sum() > 0, "Train split should contain positive examples"
    assert y_test.sum() > 0, "Test split should contain positive examples"
