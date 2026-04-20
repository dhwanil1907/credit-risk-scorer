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


def make_bureau_df():
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 2, 3, 3],
            "SK_ID_BUREAU": [101, 102, 201, 301, 302],
            "CREDIT_ACTIVE": ["Active", "Closed", "Active", "Active", "Active"],
            "CREDIT_DAY_OVERDUE": [0, 5, 10, 0, 3],
            "AMT_CREDIT_SUM_DEBT": [1000.0, 200.0, 500.0, 800.0, 100.0],
            "AMT_CREDIT_SUM": [5000.0, 3000.0, 2000.0, 4000.0, 1500.0],
            "AMT_CREDIT_SUM_LIMIT": [6000.0, 3500.0, 2500.0, 4500.0, 2000.0],
            "AMT_CREDIT_SUM_OVERDUE": [0.0, 10.0, 0.0, 0.0, 5.0],
            "DAYS_CREDIT": [-100, -200, -300, -400, -500],
            "CREDIT_TYPE": ["Consumer credit"] * 5,
            "AMT_ANNUITY": [100.0, 200.0, 150.0, 120.0, 80.0],
            "AMT_CREDIT_MAX_OVERDUE": [0.0, 5.0, 0.0, 0.0, 2.0],
            "CNT_CREDIT_PROLONG": [0, 1, 0, 0, 0],
        }
    )


def make_application_df():
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3, 4, 5],
            "TARGET": [0, 1, 0, 0, 1],
            "DAYS_EMPLOYED": [-500, 365243, -200, 365243, -1000],
            "DAYS_BIRTH": [-10000, -12000, -15000, -8000, -20000],
            "AMT_CREDIT": [100000.0, 200000.0, 150000.0, 80000.0, 300000.0],
            "AMT_INCOME_TOTAL": [50000.0, None, 75000.0, 30000.0, 100000.0],
            "AMT_ANNUITY": [5000.0, 10000.0, None, 4000.0, 15000.0],
            "EXT_SOURCE_1": [0.5, None, 0.3, 0.7, 0.4],
            "EXT_SOURCE_2": [0.6, 0.4, None, 0.8, 0.3],
            "EXT_SOURCE_3": [0.7, 0.5, 0.2, None, 0.6],
            "CODE_GENDER": ["M", "F", "XNA", "M", "F"],
            "NAME_CONTRACT_TYPE": [
                "Cash loans",
                "Revolving loans",
                "Cash loans",
                "Cash loans",
                "Revolving loans",
            ],
            "FLAG_OWN_CAR": ["Y", "N", "Y", "N", "Y"],
            "FLAG_OWN_REALTY": ["Y", "N", "N", "Y", "Y"],
        }
    )


def test_load_bureau_aggregates_shape():
    bureau_df = make_bureau_df()
    result = load_bureau_aggregates(bureau_df)
    assert result.shape[0] == 3, "Should have one row per unique SK_ID_CURR"
    expected_cols = {
        "bureau_count",
        "bureau_active_count",
        "bureau_closed_count",
        "bureau_max_overdue",
        "bureau_mean_overdue",
        "bureau_total_debt",
        "bureau_avg_credit",
        "bureau_max_line_credit",
        "bureau_sum_limit",
        "bureau_sum_overdue_amt",
        "bureau_mean_days_credit",
        "bureau_credit_type_nunique",
        "bureau_mean_line_annuity",
        "bureau_max_max_overdue",
        "bureau_cnt_prolong_sum",
    }
    assert expected_cols.issubset(result.columns), "Missing aggregate columns"


def test_load_previous_application_aggregates_shape() -> None:
    prev = pd.DataFrame(
        {
            "SK_ID_PREV": [10, 11, 12],
            "SK_ID_CURR": [1, 1, 2],
            "AMT_CREDIT": [1000.0, 2000.0, 3000.0],
            "AMT_APPLICATION": [900.0, 1900.0, 2800.0],
            "AMT_ANNUITY": [100.0, 200.0, 300.0],
            "AMT_DOWN_PAYMENT": [50.0, 0.0, 100.0],
        }
    )
    out = load_previous_application_aggregates(prev)
    assert {"SK_ID_CURR", "prev_app_count", "prev_amt_credit_sum"}.issubset(out.columns)
    assert out.loc[out["SK_ID_CURR"] == 1, "prev_app_count"].iloc[0] == 2


def test_clean_application_handles_days_employed_anomaly():
    df = make_application_df().copy()
    cleaned = clean_application(df)
    assert not (cleaned["DAYS_EMPLOYED"] == 365243).any(), (
        "365243 anomaly should be replaced with NaN"
    )


def test_clean_application_imputes_nulls():
    df = make_application_df().copy()
    cleaned = clean_application(df)
    for col in ["AMT_INCOME_TOTAL", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
        assert cleaned[col].isna().sum() == 0, f"{col} should have no nulls after cleaning"


def test_encode_categoricals_no_string_columns():
    df = make_application_df().copy()
    df = clean_application(df)
    encoded = encode_categoricals(df)
    object_cols = encoded.select_dtypes(include="object").columns.tolist()
    assert len(object_cols) == 0, f"Object columns remain after encoding: {object_cols}"


def test_engineer_features_adds_ratios():
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
    rng = np.random.default_rng(42)
    n = 200
    y = np.zeros(n, dtype=int)
    pos_idx = rng.choice(n, size=16, replace=False)
    y[pos_idx] = 1
    X = pd.DataFrame(rng.random((n, 5)), columns=[f"f{i}" for i in range(5)])
    y_series = pd.Series(y)

    X_train, X_test, y_train, y_test = split_data(X, y_series)

    assert len(X_train) == 160, f"Expected 160 train rows, got {len(X_train)}"
    assert len(X_test) == 40, f"Expected 40 test rows, got {len(X_test)}"
    assert y_train.sum() > 0, "Train split should contain positive examples"
    assert y_test.sum() > 0, "Test split should contain positive examples"
