"""
Streamlit front-end: applicant inputs → same encoding/engineering as training, XGBoost risk score,
local SHAP waterfall, and static model / global-SHAP artifacts from the training run.

This is the interactive web application that a loan officer or analyst sees in their browser.
It lets the user enter an applicant's details in a sidebar, and instantly shows:
- A risk score from 0–100 (higher = safer)
- A colour-coded risk tier (Low / Medium / High Risk)
- A personalised SHAP chart explaining why the model gave that score
- A comparison table of all three model's performance metrics
- The global SHAP summary chart from the training run
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

# Use a non-interactive chart backend so matplotlib can save images without opening windows
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src.data_prep import encode_categoricals, engineer_features


def _coerce_shap_matrix(raw: object, expected_shape: tuple[int, int]) -> np.ndarray:
    """
    Normalise the SHAP library's output into a consistent matrix format.

    The SHAP library can return values in slightly different formats depending
    on the version and model type. This function handles those variations and
    always returns a clean table: one row per applicant, one column per feature,
    containing contributions toward the default class (the one we care about).

    This is identical to the same helper in src/explain.py — duplicated here
    so the web app does not depend on the explain module at runtime.
    """
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
    """
    Get the model's single baseline value to anchor the waterfall chart.

    The baseline is the model's average predicted default probability across
    all training applicants — the starting point before any individual features
    are taken into account. When a binary model returns two baseline values
    (one per class), we pick the one for the default class (index 1).
    """
    ev_arr = np.asarray(explainer.expected_value).ravel()
    if ev_arr.size > 1:
        return float(ev_arr[1])
    return float(ev_arr[0])


@st.cache_resource
def load_models() -> dict[str, object]:
    """
    Load all three trained model files from disk into memory.

    This function is cached — Streamlit will only load the models once when the
    app first starts, then reuse them for every subsequent interaction. This
    keeps the app responsive; loading large model files on every click would
    make the app feel slow.

    Returns a dictionary with model names as keys and fitted model objects as values.
    """
    return {
        "xgboost": joblib.load(_ROOT / "models" / "xgboost.pkl"),
        "random_forest": joblib.load(_ROOT / "models" / "random_forest.pkl"),
        "logistic_regression": joblib.load(_ROOT / "models" / "logistic_regression.pkl"),
    }


@st.cache_resource
def xgb_shap_explainer() -> shap.TreeExplainer:
    """
    Load and cache the SHAP explainer for the XGBoost model.

    The SHAP explainer analyses the internal structure of the XGBoost model's
    decision trees. Building this explainer is computationally expensive, so
    we cache it so it is only built once per server session — not on every
    applicant update.
    """
    # TreeExplainer is expensive to construct; cache once per Streamlit server process
    return shap.TreeExplainer(load_models()["xgboost"])


def _waterfall_figure(X_row: pd.DataFrame) -> plt.Figure:
    """
    Generate a personalised SHAP waterfall chart for the current applicant.

    A waterfall chart shows exactly how the model arrived at its prediction for
    this one applicant. Starting from the average baseline probability, each bar
    shows how much one feature pushed the score up (toward default) or down
    (toward safe). The final position is the model's predicted default probability.

    This mirrors the same chart produced by src/explain.py — but computed live
    in the web app for whatever values the user has entered in the sidebar.
    """
    explainer = xgb_shap_explainer()

    # Calculate SHAP contributions for this single applicant row
    raw = explainer.shap_values(X_row)
    shap_matrix = _coerce_shap_matrix(raw, (X_row.shape[0], X_row.shape[1]))
    vals = shap_matrix[0]

    # Get the average baseline prediction to anchor the waterfall
    base = _expected_value_scalar(explainer)

    # Package into the format the SHAP waterfall chart expects
    explanation = shap.Explanation(
        values=vals,                           # Per-feature SHAP contributions
        base_values=base,                      # Baseline (average) prediction
        data=X_row.iloc[0].to_numpy(),         # The actual feature values for this applicant
        feature_names=list(X_row.columns),     # Column names for chart labels
    )

    # Draw the chart (show=False = render to memory, not a pop-up window)
    shap.plots.waterfall(explanation, show=False)
    fig = plt.gcf()
    return fig


@st.cache_data
def load_feature_cols() -> list[str]:
    """
    Load the saved list of feature column names used during training.

    When the model was trained, the exact list of columns (and their order)
    was saved to outputs/feature_cols.json. We load that same list here to
    guarantee the web app sends the model data in the exact same column order
    it was trained on. Using the wrong column order would produce nonsensical
    predictions.

    Cached so the file is only read from disk once per session.
    """
    path = _ROOT / "outputs" / "feature_cols.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_metrics_csv() -> pd.DataFrame:
    """
    Load the model performance comparison table saved by train.py.

    This table was created when the models were trained and contains metrics
    (ROC-AUC, F1, precision, recall) for all three models on the held-out
    test set. It is displayed in the app so users can see how each model
    compares. Cached so the file is only read from disk once per session.
    """
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
    Convert the values entered in the sidebar into a single-row table the model can read.

    The model was trained on a specific table format with dozens of columns.
    This function takes the user's inputs from the sidebar and builds a row in
    that exact format — filling in sensible neutral defaults for any fields not
    collected in the sidebar (e.g. regional rating is set to the average of 2).

    Notes on defaults:
    - Fields not shown in the sidebar use neutral/average values so they don't
      skew the prediction in either direction
    - Bureau fields not collected (e.g. total debt) default to 0, which is the
      same value applicants with no bureau records receive during training
    - The model uses days for age and employment (negative numbers mean "in the
      past"), so we convert from years back to days internally
    """
    # Convert years employed to negative days (the format used in training data)
    # If years_employed is 0, use -30 days as a safe non-zero placeholder
    days_employed = -int(years_employed * 365) if years_employed > 0 else -30

    return pd.DataFrame(
        [
            {
                "NAME_CONTRACT_TYPE": contract,          # "Cash loans" or "Revolving loans"
                "CODE_GENDER": gender,                   # "M" or "F"
                "FLAG_OWN_CAR": own_car,                 # "Y" or "N"
                "FLAG_OWN_REALTY": own_realty,           # "Y" or "N"
                "CNT_CHILDREN": int(children),
                # Family size = children + 1 parent (minimum 2, since applicant counts as 1)
                "CNT_FAM_MEMBERS": max(int(children) + 1, 2),
                "AMT_INCOME_TOTAL": float(income),
                "AMT_CREDIT": float(loan),
                "AMT_ANNUITY": float(annuity),
                "AMT_GOODS_PRICE": float(goods_price),
                "DAYS_BIRTH": -int(age_years * 365),     # Negative: days before the application
                "DAYS_EMPLOYED": days_employed,           # Negative: days before the application
                "EXT_SOURCE_1": 0.5,                     # External credit score 1 — neutral default
                "EXT_SOURCE_2": float(ext2),             # External credit score 2 — from sidebar
                "EXT_SOURCE_3": float(ext3),             # External credit score 3 — from sidebar
                "REGION_POPULATION_RELATIVE": 0.02,      # Regional density — neutral default
                "DAYS_ID_PUBLISH": -4000,                # Days since ID was updated — neutral default
                "OWN_CAR_AGE": 5.0 if own_car == "Y" else 0.0,  # Car age: 5 years if they own one
                "REGION_RATING_CLIENT": 2,               # Region quality rating — neutral default (1–3 scale)
                "REGION_RATING_CLIENT_W_CITY": 2,        # City-weighted region rating — neutral default
                "bureau_count": int(bureau_count),       # Number of prior credit records — from sidebar
                "bureau_active_count": 0,                # Active credit lines — defaulted to 0
                "bureau_max_overdue": int(bureau_max_overdue),  # Worst overdue days — from sidebar
                "bureau_total_debt": 0.0,                # Total outstanding debt — defaulted to 0
                "bureau_avg_credit": 0.0,                # Average credit line size — defaulted to 0
            }
        ]
    )


def preprocess_for_model(raw_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Apply the same data transformations to sidebar inputs as were applied during training.

    Consistency is critical: the model was trained on data that had been through
    specific steps (text → numbers, calculated ratios, etc.). If we send the model
    raw sidebar values without those same steps, it will interpret the numbers
    incorrectly and produce wrong predictions.

    Steps applied here:
    1. encode_categoricals: convert text like "M"/"F" to 1/0
    2. engineer_features: add calculated columns like "loan as % of income"
    3. Fill any missing columns with 0 (for optional features not collected in the sidebar)
    4. Select and reorder columns to exactly match what the model was trained on
    """
    # Convert text fields to numbers (same mappings as training)
    df = encode_categoricals(raw_df)

    # Add the derived/calculated columns (same formulas as training)
    df = engineer_features(df)

    # Any feature column that couldn't be produced from sidebar inputs gets set to 0
    # (same default as applicants with no data for that feature in training)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    # Select only the columns the model knows about, in the exact same order as training
    out = df[feature_cols].astype(np.float64)
    return out


def main() -> None:
    """
    Build and display the web application.

    Streamlit re-runs this entire function from top to bottom every time the
    user interacts with the sidebar — so the score and charts update instantly
    with every change.
    """
    st.set_page_config(page_title="Credit Risk Scorer", layout="wide")

    # Load all resources (cached — only happens once per server session)
    models = load_models()
    xgb = models["xgboost"]
    _ = xgb_shap_explainer()       # Pre-warm the cache so first prediction is fast
    feature_cols = load_feature_cols()
    metrics_df = load_metrics_csv()

    # ──────────────────────────────────────────────────────────────────────────
    # SIDEBAR: Input fields for the applicant's details
    # The user fills these in; every change triggers a new prediction instantly
    # ──────────────────────────────────────────────────────────────────────────
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

    # ──────────────────────────────────────────────────────────────────────────
    # SCORING: Turn sidebar values into a prediction
    # ──────────────────────────────────────────────────────────────────────────

    # Build a raw data row from the sidebar inputs
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

    # Apply the same preprocessing steps used during training
    X = preprocess_for_model(raw, feature_cols)

    # Get the model's predicted probability of default (a number between 0 and 1)
    p_default = float(xgb.predict_proba(X)[0, 1])

    # Convert to a 0–100 risk score where higher = safer (invert the default probability)
    # e.g. 8% default probability → risk score of 92 (very safe)
    # e.g. 60% default probability → risk score of 40 (high risk)
    risk_score = int(round((1.0 - p_default) * 100))
    risk_score = max(0, min(100, risk_score))  # Clamp to [0, 100] just in case

    # Assign a risk tier and display colour based on the score
    if risk_score >= 70:
        tier, color = "Low Risk", "#1a7f37"       # Green: safe applicant
    elif risk_score >= 40:
        tier, color = "Medium Risk", "#c27f00"    # Amber: caution — borderline applicant
    else:
        tier, color = "High Risk", "#c41e3a"      # Red: high probability of default

    # ──────────────────────────────────────────────────────────────────────────
    # MAIN PAGE: Display score (left column) and SHAP chart (right column)
    # ──────────────────────────────────────────────────────────────────────────
    left, right = st.columns(2)

    with left:
        # Show the numeric score, the risk tier label, a progress bar, and the raw probability
        st.metric("Risk score", f"{risk_score} / 100")
        st.markdown(
            f'<p style="color:{color};font-weight:700;font-size:1.1rem;">{tier}</p>',
            unsafe_allow_html=True,
        )
        st.progress(risk_score / 100.0)
        st.caption(f"Raw default probability: {p_default * 100:.2f}%")

    with right:
        # Show the personalised SHAP explanation chart for this applicant
        st.subheader("SHAP — this applicant")
        wf_fig = _waterfall_figure(X)
        st.pyplot(wf_fig, clear_figure=True)
        plt.close(wf_fig)

    # ──────────────────────────────────────────────────────────────────────────
    # LOWER SECTION: Model comparison table and global SHAP summary
    # ──────────────────────────────────────────────────────────────────────────
    st.divider()

    # Display the performance metrics table for all three models
    st.subheader("Model comparison (holdout metrics)")
    display_metrics = metrics_df.drop(columns=["confusion_matrix"], errors="ignore")
    numeric_cols = display_metrics.select_dtypes(include=[np.number]).columns
    formatted = display_metrics.copy()
    # Format all numbers to 4 decimal places for readability
    for c in numeric_cols:
        formatted[c] = formatted[c].map(lambda v: f"{float(v):.4f}")
    st.dataframe(formatted, hide_index=True, use_container_width=True)

    # Display the global SHAP summary chart generated by src/explain.py
    # This chart shows which features matter most across all applicants (not just this one)
    st.subheader("Global SHAP (training sample)")
    summary_path = _ROOT / "outputs" / "shap_summary.png"
    if summary_path.is_file():
        st.image(str(summary_path), use_container_width=True)
    else:
        # Guide the user to generate the chart if it doesn't exist yet
        st.warning("Run `python src/explain.py` to generate outputs/shap_summary.png.")


if __name__ == "__main__":
    main()
