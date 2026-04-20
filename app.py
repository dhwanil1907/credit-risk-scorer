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


def _compute_shap(X_row: pd.DataFrame) -> tuple[np.ndarray, float]:
    """
    Return (shap_values_1d, base_value) for a single-row DataFrame.

    shap_values_1d[i] > 0 means feature i pushes toward default (increases risk).
    shap_values_1d[i] < 0 means feature i pushes toward repayment (decreases risk).
    """
    explainer = xgb_shap_explainer()
    raw = explainer.shap_values(X_row)
    shap_matrix = _coerce_shap_matrix(raw, (X_row.shape[0], X_row.shape[1]))
    return shap_matrix[0], _expected_value_scalar(explainer)


def _waterfall_figure(X_row: pd.DataFrame, vals: np.ndarray, base: float) -> plt.Figure:
    """
    Generate a personalised SHAP waterfall chart for the current applicant.

    A waterfall chart shows exactly how the model arrived at its prediction for
    this one applicant. Starting from the average baseline probability, each bar
    shows how much one feature pushed the score up (toward default) or down
    (toward safe). The final position is the model's predicted default probability.
    """
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


def apply_guardrails(model_score: int, X_row: pd.DataFrame) -> tuple[int, list[str]]:
    """
    Apply business rule overrides on top of the model score.

    In production credit systems, a statistical model sits underneath a policy
    rules engine that enforces hard limits mandated by risk teams or regulators.
    Rules are evaluated in order. The harshest cap wins.

    Returns:
        adjusted_score: the final score after all rules are applied
        triggered: list of human-readable descriptions of rules that fired
    """
    score = model_score
    triggered: list[str] = []

    annuity_ratio = (
        float(X_row["ANNUITY_INCOME_RATIO"].iloc[0])
        if "ANNUITY_INCOME_RATIO" in X_row.columns
        else 0.0
    )
    credit_ratio = (
        float(X_row["CREDIT_INCOME_RATIO"].iloc[0]) if "CREDIT_INCOME_RATIO" in X_row.columns else 0.0
    )
    max_overdue = (
        float(X_row["bureau_max_overdue"].iloc[0]) if "bureau_max_overdue" in X_row.columns else 0.0
    )

    if annuity_ratio > 1.0:
        if score > 30:
            score = 30
        triggered.append(
            f"Repayments exceed annual income (ratio: {annuity_ratio:.2f}x) — capped at 30"
        )
    elif annuity_ratio > 0.5:
        if score > 55:
            score = 55
        triggered.append(
            f"Repayments exceed 50% of income (ratio: {annuity_ratio:.2f}x) — capped at 55"
        )

    if credit_ratio > 10.0:
        if score > 45:
            score = 45
        triggered.append(f"Loan is {credit_ratio:.1f}× annual income — capped at 45")

    if max_overdue > 60:
        score = max(0, score - 15)
        triggered.append(f"Bureau record shows {int(max_overdue)} days overdue — score reduced by 15")

    return score, triggered


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
    # SIDEBAR: Inputs grouped by impact level
    # The three external credit scores are the strongest predictors — put them
    # first and label them clearly so users know where to look for big changes.
    # ──────────────────────────────────────────────────────────────────────────
    st.sidebar.title("Applicant Profile")

    st.sidebar.markdown("### Credit Scores *(biggest impact)*")
    st.sidebar.caption("Moving these sliders will produce the largest changes in the risk score.")
    ext1 = st.sidebar.slider("External score 1", 0.0, 1.0, 0.5, 0.01)
    ext2 = st.sidebar.slider("External score 2", 0.0, 1.0, 0.5, 0.01)
    ext3 = st.sidebar.slider("External score 3", 0.0, 1.0, 0.5, 0.01)

    st.sidebar.markdown("### Financial Details")
    income = st.sidebar.number_input("Annual income ($)", min_value=1.0, value=180_000.0, step=1000.0)
    loan = st.sidebar.number_input("Loan amount ($)", min_value=1.0, value=250_000.0, step=1000.0)
    annuity = st.sidebar.number_input("Annual repayment / annuity ($)", min_value=0.0, value=15_000.0, step=500.0)
    goods_price = st.sidebar.number_input("Goods price ($)", min_value=0.0, value=250_000.0, step=1000.0)
    contract = st.sidebar.selectbox("Contract type", ["Cash loans", "Revolving loans"], index=0)

    st.sidebar.markdown("### Personal Details")
    age_years = st.sidebar.slider("Age (years)", 18, 70, 35)
    years_employed = st.sidebar.slider("Years employed", 0, 40, 5)
    gender = st.sidebar.selectbox("Gender", ["M", "F"], index=0)
    own_car = st.sidebar.selectbox("Owns car", ["Y", "N"], index=0)
    own_realty = st.sidebar.selectbox("Owns realty", ["Y", "N"], index=0)
    children = st.sidebar.slider("Number of children", 0, 10, 0)

    st.sidebar.markdown("### Credit History")
    bureau_count = st.sidebar.slider("Bureau record count", 0, 50, 3)
    bureau_max_overdue = st.sidebar.slider("Bureau max days overdue", 0, 120, 0)

    # ──────────────────────────────────────────────────────────────────────────
    # SCORING
    # ──────────────────────────────────────────────────────────────────────────
    raw = _build_raw_applicant_frame(
        contract, gender, own_car, own_realty, children,
        income, loan, annuity, goods_price,
        age_years, years_employed,
        ext2, ext3, bureau_count, bureau_max_overdue,
    )
    # Override EXT_SOURCE_1 with the sidebar value (was previously hardcoded to 0.5)
    raw["EXT_SOURCE_1"] = float(ext1)

    X = preprocess_for_model(raw, feature_cols)

    p_default = float(xgb.predict_proba(X)[0, 1])
    model_score = max(0, min(100, int(round((1.0 - p_default) * 100))))
    risk_score, triggered_rules = apply_guardrails(model_score, X)

    if risk_score >= 70:
        tier, bg_color, text_color = "Low Risk", "#d4edda", "#1a7f37"
    elif risk_score >= 40:
        tier, bg_color, text_color = "Medium Risk", "#fff3cd", "#856404"
    else:
        tier, bg_color, text_color = "High Risk", "#f8d7da", "#842029"

    # Compute SHAP once; reuse for both the factor panel and the waterfall chart
    shap_vals, shap_base = _compute_shap(X)

    # ──────────────────────────────────────────────────────────────────────────
    # HEADER
    # ──────────────────────────────────────────────────────────────────────────
    st.title("Credit Risk Assessment")
    st.caption("Adjust the applicant's details in the sidebar — the score and explanation update instantly.")
    st.divider()

    # ──────────────────────────────────────────────────────────────────────────
    # ROW 1: Score card + key drivers
    # ──────────────────────────────────────────────────────────────────────────
    score_col, drivers_col = st.columns([1, 2], gap="large")

    with score_col:
        st.markdown(
            f"""
            <div style="
                background:{bg_color};
                border-left: 6px solid {text_color};
                border-radius: 8px;
                padding: 28px 24px;
                text-align: center;
            ">
                <div style="font-size:0.9rem;color:#555;margin-bottom:4px;letter-spacing:0.05em;">
                    RISK SCORE
                </div>
                <div style="font-size:4rem;font-weight:800;color:{text_color};line-height:1;">
                    {risk_score}
                </div>
                <div style="font-size:0.85rem;color:#777;margin-bottom:12px;">out of 100</div>
                <div style="
                    display:inline-block;
                    background:{text_color};
                    color:white;
                    padding:4px 16px;
                    border-radius:20px;
                    font-weight:700;
                    font-size:0.95rem;
                ">
                    {tier}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("")
        st.progress(risk_score / 100.0)
        if model_score != risk_score:
            st.caption(
                f"Model score: **{model_score}** → Policy-adjusted: **{risk_score}** "
                f"(default probability: {p_default * 100:.1f}%)"
            )
        else:
            st.caption(
                f"Default probability: **{p_default * 100:.1f}%** "
                f"({'likely to repay' if p_default < 0.5 else 'likely to default'})"
            )

    with drivers_col:
        st.markdown("#### What's driving this score?")
        st.caption(
            "Positive factors reduce risk (push score up). "
            "Negative factors increase risk (push score down)."
        )

        # Build a ranked factor table from SHAP values.
        # Positive SHAP → pushes toward default → LOWERS the risk score.
        # We flip the sign so positive = good for the applicant.
        feature_names = list(X.columns)
        factor_df = pd.DataFrame({
            "Feature": feature_names,
            "Impact": -shap_vals,   # flipped: positive = reduces risk
        })

        # Friendly display names for the most common features
        label_map = {
            "EXT_SOURCE_1": "External credit score 1",
            "EXT_SOURCE_2": "External credit score 2",
            "EXT_SOURCE_3": "External credit score 3",
            "CREDIT_INCOME_RATIO": "Loan-to-income ratio",
            "ANNUITY_INCOME_RATIO": "Repayment burden",
            "AGE_YEARS": "Age",
            "YEARS_EMPLOYED": "Years employed",
            "AMT_CREDIT": "Loan amount",
            "AMT_INCOME_TOTAL": "Annual income",
            "bureau_max_overdue": "Worst overdue days",
            "bureau_count": "Bureau record count",
            "CREDIT_TERM": "Loan term (months)",
        }
        factor_df["Label"] = factor_df["Feature"].map(label_map).fillna(factor_df["Feature"])
        factor_df = factor_df.sort_values("Impact", ascending=False)

        top_positive = factor_df[factor_df["Impact"] > 0].head(4)
        top_negative = factor_df[factor_df["Impact"] < 0].head(4)

        left_f, right_f = st.columns(2)
        with left_f:
            st.markdown("**Reducing risk (good)**")
            if top_positive.empty:
                st.caption("None significant")
            for _, row in top_positive.iterrows():
                bar_pct = min(int(abs(row["Impact"]) * 600), 100)
                st.markdown(
                    f'<div style="margin-bottom:6px;">'
                    f'<span style="font-size:0.85rem;">{row["Label"]}</span><br>'
                    f'<div style="background:#d4edda;width:{bar_pct}%;height:8px;'
                    f'border-radius:4px;display:inline-block;"></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        with right_f:
            st.markdown("**Increasing risk (concern)**")
            if top_negative.empty:
                st.caption("None significant")
            for _, row in top_negative.iterrows():
                bar_pct = min(int(abs(row["Impact"]) * 600), 100)
                st.markdown(
                    f'<div style="margin-bottom:6px;">'
                    f'<span style="font-size:0.85rem;">{row["Label"]}</span><br>'
                    f'<div style="background:#f8d7da;width:{bar_pct}%;height:8px;'
                    f'border-radius:4px;display:inline-block;"></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.divider()

    if triggered_rules:
        st.warning(
            "**Policy rules triggered:**\n" + "\n".join(f"- {r}" for r in triggered_rules)
        )

    # ──────────────────────────────────────────────────────────────────────────
    # ROW 2: SHAP waterfall (full detail explanation)
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown("#### Full explanation — how the model built this score")
    st.caption(
        "Each bar shows one feature's contribution. "
        "Red bars push toward default (lower score); blue bars push toward repayment (higher score)."
    )
    wf_fig = _waterfall_figure(X, shap_vals, shap_base)
    st.pyplot(wf_fig, clear_figure=True)
    plt.close(wf_fig)

    st.divider()

    # ──────────────────────────────────────────────────────────────────────────
    # ROW 3: Model comparison + global SHAP (collapsed by default)
    # ──────────────────────────────────────────────────────────────────────────
    with st.expander("Model comparison — holdout metrics", expanded=False):
        display_metrics = metrics_df.drop(columns=["confusion_matrix"], errors="ignore")
        numeric_cols = display_metrics.select_dtypes(include=[np.number]).columns
        formatted = display_metrics.copy()
        for c in numeric_cols:
            formatted[c] = formatted[c].map(lambda v: f"{float(v):.4f}")
        st.dataframe(formatted, hide_index=True, use_container_width=True)

    with st.expander("Global feature importance (across all training applicants)", expanded=False):
        summary_path = _ROOT / "outputs" / "shap_summary.png"
        if summary_path.is_file():
            st.image(str(summary_path), use_container_width=True)
        else:
            st.warning("Run `python src/explain.py` to generate outputs/shap_summary.png.")


if __name__ == "__main__":
    main()
