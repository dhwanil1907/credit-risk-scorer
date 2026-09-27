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

from src.applicant import build_raw_applicant_frame, preprocess_for_model
from src.guardrails import apply_guardrails
from src.labels import feature_label
from src.shap_utils import coerce_shap_matrix, expected_value_scalar

_ROOT = Path(__file__).resolve().parent.parent


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
    gender: str = "M"
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
    ) -> None:
        model_path = model_path or (_ROOT / "models" / "xgboost.pkl")
        feature_cols_path = feature_cols_path or (_ROOT / "outputs" / "feature_cols.json")
        self.model = joblib.load(model_path)
        with open(feature_cols_path, encoding="utf-8") as handle:
            self.feature_cols: list[str] = json.load(handle)
        self.model_columns = model_feature_names(self.model)
        self.explainer = shap.TreeExplainer(self.model)

    def score(self, applicant: Applicant) -> ScoreResult:
        contract = "Cash loans" if applicant.loan_type == "cash" else "Revolving loans"
        raw = build_raw_applicant_frame(
            contract,
            applicant.gender,
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
        default_probability = float(self.model.predict_proba(model_row)[0, 1])
        model_score = max(0, min(100, int(round((1.0 - default_probability) * 100))))
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
