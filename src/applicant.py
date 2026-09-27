"""Turn an applicant profile into the same feature row used at training time."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_prep import encode_categoricals, engineer_features


def build_raw_applicant_frame(
    contract: str,
    gender: str,
    own_car: str,
    own_realty: str,
    children: int,
    income: float,
    loan: float,
    annuity: float,
    goods_price: float,
    age_years: int,
    years_employed: int,
    ext2: float,
    ext3: float,
    bureau_count: int,
    bureau_max_overdue: int,
    ext1: float = 0.5,
) -> pd.DataFrame:
    """
    Build one applicant row. Fields the form does not collect use the same
    neutral defaults as training (no bureau history → 0).
    """
    days_employed = -int(years_employed * 365) if years_employed > 0 else -30
    return pd.DataFrame(
        [
            {
                "NAME_CONTRACT_TYPE": contract,
                "CODE_GENDER": gender,
                "FLAG_OWN_CAR": own_car,
                "FLAG_OWN_REALTY": own_realty,
                "CNT_CHILDREN": int(children),
                "CNT_FAM_MEMBERS": max(int(children) + 1, 2),
                "AMT_INCOME_TOTAL": float(income),
                "AMT_CREDIT": float(loan),
                "AMT_ANNUITY": float(annuity),
                "AMT_GOODS_PRICE": float(goods_price),
                "DAYS_BIRTH": -int(age_years * 365),
                "DAYS_EMPLOYED": days_employed,
                "EXT_SOURCE_1": float(ext1),
                "EXT_SOURCE_2": float(ext2),
                "EXT_SOURCE_3": float(ext3),
                "REGION_POPULATION_RELATIVE": 0.02,
                "DAYS_ID_PUBLISH": -4000,
                "OWN_CAR_AGE": 5.0 if own_car == "Y" else 0.0,
                "REGION_RATING_CLIENT": 2,
                "REGION_RATING_CLIENT_W_CITY": 2,
                "bureau_count": int(bureau_count),
                "bureau_active_count": 0,
                "bureau_max_overdue": int(bureau_max_overdue),
                "bureau_total_debt": 0.0,
                "bureau_avg_credit": 0.0,
            }
        ]
    )


def preprocess_for_model(raw_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Encode, engineer features, and order columns to match training."""
    df = encode_categoricals(raw_df)
    df = engineer_features(df)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    return df[feature_cols].astype(np.float64)
