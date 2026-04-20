"""
Streamlit front-end: applicant inputs → same encoding/engineering as training, XGBoost risk score,
local SHAP waterfall, and static model / global-SHAP artifacts from the training run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Repo root (this file lives at project root next to `src/` and `models/`).
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src.data_prep import encode_categoricals, engineer_features


def _coerce_shap_matrix(raw: object, expected_shape: tuple[int, int]) -> np.ndarray:
    """Same coercion rules as src.explain (binary XGB margin / list outputs)."""
    if isinstance(raw, list):
        if len(raw) < 2:
            raise ValueError("Unexpected SHAP list length for binary classification.")
        arr = np.asarray(raw[1])
    else:
        arr = np.asarray(raw)
    if arr.ndim == 3 and arr.shape[-1] == 2:
        arr = arr[:, :, 1]
    if arr.shape != expected_shape:
        raise ValueError(f"SHAP values shape {arr.shape} does not match X {expected_shape}.")
    return arr


def _expected_value_scalar(explainer: shap.TreeExplainer) -> float:
    """Match TreeExplainer margin output to a single baseline for the waterfall (binary)."""
    ev_arr = np.asarray(explainer.expected_value).ravel()
    if ev_arr.size > 1:
        return float(ev_arr[1])
    return float(ev_arr[0])


@st.cache_resource
def load_models() -> dict[str, object]:
    """Load fitted estimators once per process; joblib blobs stay in memory across reruns."""
    return {
        "xgboost": joblib.load(_ROOT / "models" / "xgboost.pkl"),
        "random_forest": joblib.load(_ROOT / "models" / "random_forest.pkl"),
        "logistic_regression": joblib.load(_ROOT / "models" / "logistic_regression.pkl"),
    }


@st.cache_resource
def xgb_shap_explainer() -> shap.TreeExplainer:
    """TreeExplainer is expensive to construct; cache once per Streamlit server process."""
    return shap.TreeExplainer(load_models()["xgboost"])


def _waterfall_figure(X_row: pd.DataFrame) -> plt.Figure:
    """Build SHAP waterfall for one aligned feature row (mirrors src/explain.plot_shap_waterfall)."""
    explainer = xgb_shap_explainer()
    raw = explainer.shap_values(X_row)
    shap_matrix = _coerce_shap_matrix(raw, (X_row.shape[0], X_row.shape[1]))
    vals = shap_matrix[0]
    base = _expected_value_scalar(explainer)
    explanation = shap.Explanation(
        values=vals,
        base_values=base,
        data=X_row.iloc[0].to_numpy(),
        feature_names=list(X_row.columns),
    )
    shap.plots.waterfall(explanation, show=False)
    fig = plt.gcf()
    return fig


@st.cache_data
def load_feature_cols() -> list[str]:
    """Exact column order used at train time (subset of FEATURE_COLS if schema drifted)."""
    path = _ROOT / "outputs" / "feature_cols.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_metrics_csv() -> pd.DataFrame:
    """Per-model holdout scores written by train.py (static comparison in the UI)."""
    return pd.read_csv(_ROOT / "outputs" / "metrics.csv")


def _build_raw_applicant_frame(
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
) -> pd.DataFrame:
    """
    One-row application-level frame *before* encode/engineer.

    Columns absent from the sidebar use neutral training-like defaults so ratios stay finite;
    bureau fields not collected default to 0 (same as applicants with no bureau rows).
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
                "EXT_SOURCE_1": 0.5,
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
    """
    Mirror training transforms requested for the UI: label maps + engineered ratios.

    Any column listed in feature_cols but not produced here is filled with 0 so the row
    matches the matrix the models were fit on.
    """
    df = encode_categoricals(raw_df)
    df = engineer_features(df)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    out = df[feature_cols].astype(np.float64)
    return out


def main() -> None:
    st.set_page_config(page_title="Credit Risk Scorer", layout="wide")
    models = load_models()
    xgb = models["xgboost"]
    _ = xgb_shap_explainer()
    feature_cols = load_feature_cols()
    metrics_df = load_metrics_csv()

    st.sidebar.header("Applicant profile")
    contract = st.sidebar.selectbox("Contract type", ["Cash loans", "Revolving loans"], index=0)
    gender = st.sidebar.selectbox("Gender", ["M", "F"], index=0)
    own_car = st.sidebar.selectbox("Owns car", ["Y", "N"], index=0)
    own_realty = st.sidebar.selectbox("Owns realty", ["Y", "N"], index=0)
    children = st.sidebar.slider("Number of children", 0, 10, 0)
    income = st.sidebar.number_input("Annual income", min_value=1.0, value=180_000.0, step=1000.0)
    loan = st.sidebar.number_input("Loan amount", min_value=1.0, value=250_000.0, step=1000.0)
    annuity = st.sidebar.number_input("Annual repayment (annuity)", min_value=0.0, value=15_000.0, step=500.0)
    goods_price = st.sidebar.number_input("Goods price", min_value=0.0, value=250_000.0, step=1000.0)
    age_years = st.sidebar.slider("Age (years)", 18, 70, 35)
    years_employed = st.sidebar.slider("Years employed", 0, 40, 5)
    ext2 = st.sidebar.slider("External score 2", 0.0, 1.0, 0.5, 0.01)
    ext3 = st.sidebar.slider("External score 3", 0.0, 1.0, 0.5, 0.01)
    bureau_count = st.sidebar.slider("Bureau record count", 0, 50, 3)
    bureau_max_overdue = st.sidebar.slider("Bureau max days overdue", 0, 120, 0)

    raw = _build_raw_applicant_frame(
        contract,
        gender,
        own_car,
        own_realty,
        children,
        income,
        loan,
        annuity,
        goods_price,
        age_years,
        years_employed,
        ext2,
        ext3,
        bureau_count,
        bureau_max_overdue,
    )
    X = preprocess_for_model(raw, feature_cols)

    p_default = float(xgb.predict_proba(X)[0, 1])
    risk_score = int(round((1.0 - p_default) * 100))
    risk_score = max(0, min(100, risk_score))

    if risk_score >= 70:
        tier, color = "Low Risk", "#1a7f37"
    elif risk_score >= 40:
        tier, color = "Medium Risk", "#c27f00"
    else:
        tier, color = "High Risk", "#c41e3a"

    left, right = st.columns(2)
    with left:
        st.metric("Risk score", f"{risk_score} / 100")
        st.markdown(
            f'<p style="color:{color};font-weight:700;font-size:1.1rem;">{tier}</p>',
            unsafe_allow_html=True,
        )
        st.progress(risk_score / 100.0)
        st.caption(f"Raw default probability: {p_default * 100:.2f}%")

    with right:
        st.subheader("SHAP — this applicant")
        wf_fig = _waterfall_figure(X)
        st.pyplot(wf_fig, clear_figure=True)
        plt.close(wf_fig)

    st.divider()
    st.subheader("Model comparison (holdout metrics)")
    display_metrics = metrics_df.drop(columns=["confusion_matrix"], errors="ignore")
    numeric_cols = display_metrics.select_dtypes(include=[np.number]).columns
    formatted = display_metrics.copy()
    for c in numeric_cols:
        formatted[c] = formatted[c].map(lambda v: f"{float(v):.4f}")
    st.dataframe(formatted, hide_index=True, use_container_width=True)

    st.subheader("Global SHAP (training sample)")
    summary_path = _ROOT / "outputs" / "shap_summary.png"
    if summary_path.is_file():
        st.image(str(summary_path), use_container_width=True)
    else:
        st.warning("Run `python src/explain.py` to generate outputs/shap_summary.png.")


if __name__ == "__main__":
    main()
