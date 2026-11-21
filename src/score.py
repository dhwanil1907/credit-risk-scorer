"""Score one applicant with the saved XGBoost model, guardrails, and SHAP."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import shap

from src.data_prep import encode_categoricals, engineer_features
from src.guardrails import apply_guardrails
from src.shap_utils import coerce_shap_matrix, expected_value_scalar

_ROOT = Path(__file__).resolve().parent.parent

FEATURE_LABELS: dict[str, str] = {
    "EXT_SOURCE_1": "Credit bureau score #1",
    "EXT_SOURCE_2": "Credit bureau score #2",
    "EXT_SOURCE_3": "Credit bureau score #3",
    "EXT_MEAN": "Average credit bureau score",
    "CREDIT_INCOME_RATIO": "Loan size vs. income",
    "ANNUITY_INCOME_RATIO": "Yearly payments vs. income",
    "DEBT_STRESS": "Overall debt pressure",
    "EXT_MEAN_X_ANNUITY_RATIO": "Credit score × payment burden",
    "EXT_MEAN_X_CREDIT_RATIO": "Credit score × loan size",
    "AGE_YEARS": "Age",
    "YEARS_EMPLOYED": "Years at current job",
    "AMT_CREDIT": "Loan amount",
    "AMT_INCOME_TOTAL": "Annual income",
    "AMT_ANNUITY": "Yearly loan payment",
    "AMT_GOODS_PRICE": "Price of what they are buying",
    "CREDIT_TERM": "Loan length (months)",
    "bureau_max_overdue": "Worst late-payment streak (days)",
    "bureau_count": "Past loans on credit file",
    "NAME_CONTRACT_TYPE": "Loan type",
    "CNT_CHILDREN": "Number of children",
    "FLAG_OWN_CAR": "Owns a car",
    "FLAG_OWN_REALTY": "Owns a home",
    "DAYS_BIRTH": "Age (in days)",
    "DAYS_EMPLOYED": "Time at current job (in days)",
}


def feature_label(column: str) -> str:
    return FEATURE_LABELS.get(column, column.replace("_", " ").title())


def build_raw_applicant_frame(
    contract: str,
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
    Build one applicant row from exactly the fields the form collects.
    """
    days_employed = -int(years_employed * 365) if years_employed > 0 else -30
    return pd.DataFrame(
        [
            {
                "NAME_CONTRACT_TYPE": contract,
                "FLAG_OWN_CAR": own_car,
                "FLAG_OWN_REALTY": own_realty,
                "CNT_CHILDREN": int(children),
                "AMT_INCOME_TOTAL": float(income),
                "AMT_CREDIT": float(loan),
                "AMT_ANNUITY": float(annuity),
                "AMT_GOODS_PRICE": float(goods_price),
                "DAYS_BIRTH": -int(age_years * 365),
                "DAYS_EMPLOYED": days_employed,
                "EXT_SOURCE_1": float(ext1),
                "EXT_SOURCE_2": float(ext2),
                "EXT_SOURCE_3": float(ext3),
                "bureau_count": int(bureau_count),
                "bureau_max_overdue": int(bureau_max_overdue),
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


@dataclass(frozen=True)
class Applicant:
    yearly_income: float
    loan_amount: float
    yearly_payment: float
    purchase_price: float
    age: int
    years_employed: int
    bureau_score_1: float
    bureau_score_2: float
    bureau_score_3: float
    children: int = 0
    owns_car: bool = True
    owns_home: bool = True
    loan_type: str = "cash"
    past_loans: int = 3
    worst_late_days: int = 0


@dataclass(frozen=True)
class Reason:
    feature: str
    label: str
    direction: Literal["helps", "hurts"]
    impact: float


@dataclass(frozen=True)
class ScoreResult:
    score: int
    model_score: int
    band: str
    default_probability: float
    rules_applied: list[str]
    top_reasons: list[Reason]
    features: pd.DataFrame
    shap_values: np.ndarray
    shap_base: float


# A default chance at or above this value scores 0. With the 70 / 40 band cut-offs below,
# "Looks good" means roughly the average default chance (~8%) or better, and "High risk"
# means about 18% or worse.
SCORE_ZERO_AT = 0.30


def probability_to_score(default_probability: float) -> int:
    """Map a calibrated default chance to a 0-100 score (100 = safest)."""
    score = round(100.0 * (1.0 - default_probability / SCORE_ZERO_AT))
    return max(0, min(100, int(score)))


def risk_band(score: int) -> str:
    if score >= 70:
        return "Looks good"
    if score >= 40:
        return "Needs review"
    return "High risk"


def model_feature_names(model: object) -> list[str]:
    names_in = getattr(model, "feature_names_in_", None)
    if names_in is not None:
        return [str(c) for c in names_in]
    get_booster = getattr(model, "get_booster", None)
    if get_booster is not None:
        names = get_booster().feature_names
        if names:
            return [str(c) for c in names]
    raise ValueError("Could not read feature names from the loaded model.")


def align_to_model(features: pd.DataFrame, model_columns: list[str]) -> pd.DataFrame:
    return features.reindex(columns=model_columns, fill_value=0.0).astype(np.float64)


def top_reasons(columns: list[str], shap_values: np.ndarray, n: int = 3) -> list[Reason]:
    """Largest absolute SHAP contributions. Positive SHAP hurts the safety score."""
    order = np.argsort(-np.abs(shap_values))[:n]
    reasons: list[Reason] = []
    for index in order:
        impact = float(shap_values[int(index)])
        column = columns[int(index)]
        reasons.append(
            Reason(
                feature=column,
                label=feature_label(column),
                direction="hurts" if impact > 0 else "helps",
                impact=round(impact, 6),
            )
        )
    return reasons


class Scorer:
    """Loads the model and TreeExplainer once, then scores many applicants."""

    def __init__(
        self,
        model_path: Path | None = None,
        feature_cols_path: Path | None = None,
        calibrator_path: Path | None = None,
    ) -> None:
        model_path = model_path or (_ROOT / "models" / "xgboost.pkl")
        feature_cols_path = feature_cols_path or (_ROOT / "outputs" / "feature_cols.json")
        calibrator_path = calibrator_path or (_ROOT / "models" / "calibrator.pkl")
        self.model = joblib.load(model_path)
        self.calibrator = joblib.load(calibrator_path)
        with open(feature_cols_path, encoding="utf-8") as handle:
            self.feature_cols: list[str] = json.load(handle)
        self.model_columns = model_feature_names(self.model)
        self.explainer = shap.TreeExplainer(self.model)

    def score(self, applicant: Applicant) -> ScoreResult:
        contract = "Cash loans" if applicant.loan_type == "cash" else "Revolving loans"
        raw = build_raw_applicant_frame(
            contract,
            "Y" if applicant.owns_car else "N",
            "Y" if applicant.owns_home else "N",
            applicant.children,
            applicant.yearly_income,
            applicant.loan_amount,
            applicant.yearly_payment,
            applicant.purchase_price,
            applicant.age,
            applicant.years_employed,
            applicant.bureau_score_2,
            applicant.bureau_score_3,
            applicant.past_loans,
            applicant.worst_late_days,
            ext1=applicant.bureau_score_1,
        )
        features = preprocess_for_model(raw, self.feature_cols)
        model_row = align_to_model(features, self.model_columns)
        raw_probability = float(self.model.predict_proba(model_row)[0, 1])
        # Raw output is inflated by scale_pos_weight, so calibrate it before scoring.
        default_probability = float(self.calibrator.predict([raw_probability])[0])
        model_score = probability_to_score(default_probability)
        score, rules = apply_guardrails(model_score, features)
        raw_shap = self.explainer.shap_values(model_row)
        shap_values = coerce_shap_matrix(raw_shap, (1, model_row.shape[1]))[0]
        return ScoreResult(
            score=score,
            model_score=model_score,
            band=risk_band(score),
            default_probability=default_probability,
            rules_applied=rules,
            top_reasons=top_reasons(list(model_row.columns), shap_values),
            features=model_row,
            shap_values=shap_values,
            shap_base=expected_value_scalar(self.explainer),
        )
