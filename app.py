"""
Streamlit front-end. Scoring goes through the API (`SCORER_API_URL`, default
http://127.0.0.1:8000) so the dashboard and the service share one model path.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import httpx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

_ROOT = Path(__file__).resolve().parent
API_URL = os.environ.get("SCORER_API_URL", "http://127.0.0.1:8000").rstrip("/")

_BANDS = {
    "Looks good": ("Likely to repay if other checks pass.", "#d4edda", "#1a7f37"),
    "Needs review": ("Some red flags — worth a closer look.", "#fff3cd", "#856404"),
    "High risk": ("Strong signs they may miss payments.", "#f8d7da", "#842029"),
}


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
def load_metrics_csv() -> pd.DataFrame:
    """
    Load the model performance comparison table saved by train.py.

    This table was created when the models were trained and contains metrics
    (ROC-AUC, F1, precision, recall) for all three models on the held-out
    test set. It is displayed in the app so users can see how each model
    compares. Cached so the file is only read from disk once per session.
    """
    return pd.read_csv(_ROOT / "outputs" / "metrics.csv")


def _score_applicant(payload: dict[str, object]) -> dict[str, object]:
    """POST one applicant to the scoring API. Raises httpx.HTTPError on failure."""
    response = httpx.post(f"{API_URL}/predict", json=payload, timeout=30.0)
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict):
        raise httpx.HTTPError("Scoring API returned a non-object response.")
    return body


def main() -> None:
    """
    Build and display the web application.

    Streamlit re-runs this entire function from top to bottom every time the
    user interacts with the sidebar — so the score and charts update instantly
    with every change.
    """
    st.set_page_config(page_title="Loan Risk Checker", layout="wide", page_icon="📋")
    metrics_df = load_metrics_csv()

    st.sidebar.title("Loan applicant")
    with st.sidebar.expander("How to use this", expanded=True):
        st.markdown(
            "1. **Change any field** on the left — the score updates right away.\n\n"
            "2. **Higher score = safer** (0 = very risky, 100 = very safe).\n\n"
            "3. **Main panel** shows *why* the score changed in everyday language."
        )

    st.sidebar.markdown("### Credit scores")
    st.sidebar.caption("0 = poor history, 1 = strong history. These move the score the most.")
    ext1 = st.sidebar.slider(
        "Bureau score #1",
        0.0,
        1.0,
        0.5,
        0.01,
        help="Third-party credit rating (higher is better).",
    )
    ext2 = st.sidebar.slider("Bureau score #2", 0.0, 1.0, 0.5, 0.01, help="Another bureau rating.")
    ext3 = st.sidebar.slider("Bureau score #3", 0.0, 1.0, 0.5, 0.01, help="Another bureau rating.")

    st.sidebar.markdown("### Money")
    income = st.sidebar.number_input("Yearly income ($)", min_value=1.0, value=180_000.0, step=1000.0)
    loan = st.sidebar.number_input("Loan amount ($)", min_value=1.0, value=250_000.0, step=1000.0)
    annuity = st.sidebar.number_input(
        "Yearly payment on this loan ($)",
        min_value=0.0,
        value=15_000.0,
        step=500.0,
        help="Total they would pay toward this loan each year.",
    )
    goods_price = st.sidebar.number_input(
        "Purchase price ($)",
        min_value=0.0,
        value=250_000.0,
        step=1000.0,
        help="Cost of the item or property (often similar to loan amount).",
    )
    contract_label = st.sidebar.selectbox("Loan type", ["Fixed-term loan", "Revolving credit line"], index=0)
    contract = "Cash loans" if contract_label == "Fixed-term loan" else "Revolving loans"

    st.sidebar.markdown("### About them")
    age_years = st.sidebar.slider("Age", 18, 70, 35)
    years_employed = st.sidebar.slider("Years at current job", 0, 40, 5)
    gender = "M" if st.sidebar.selectbox("Gender", ["Male", "Female"], index=0) == "Male" else "F"
    own_car = "Y" if st.sidebar.selectbox("Owns a car?", ["Yes", "No"], index=0) == "Yes" else "N"
    own_realty = "Y" if st.sidebar.selectbox("Owns a home?", ["Yes", "No"], index=0) == "Yes" else "N"
    children = st.sidebar.slider("Children", 0, 10, 0)

    st.sidebar.markdown("### Past credit")
    bureau_count = st.sidebar.slider(
        "How many past loans on their credit file?",
        0,
        50,
        3,
        help="More records can help or hurt depending on payment history.",
    )
    bureau_max_overdue = st.sidebar.slider(
        "Worst late payment (days)",
        0,
        120,
        0,
        help="Longest they were overdue on any past loan. 0 = always on time.",
    )

    # ──────────────────────────────────────────────────────────────────────────
    # SCORING
    # ──────────────────────────────────────────────────────────────────────────
    payload = {
        "yearly_income": float(income),
        "loan_amount": float(loan),
        "yearly_payment": float(annuity),
        "purchase_price": float(goods_price),
        "age": int(age_years),
        "years_employed": int(years_employed),
        "bureau_score_1": float(ext1),
        "bureau_score_2": float(ext2),
        "bureau_score_3": float(ext3),
        "children": int(children),
        "owns_car": own_car == "Y",
        "owns_home": own_realty == "Y",
        "gender": gender,
        "loan_type": "cash" if contract == "Cash loans" else "revolving",
        "past_loans": int(bureau_count),
        "worst_late_days": int(bureau_max_overdue),
    }
    try:
        scored = _score_applicant(payload)
    except httpx.HTTPError:
        st.error(
            f"Could not reach the scoring service at {API_URL}. "
            "Start it with `uvicorn api:app --host 127.0.0.1 --port 8000`."
        )
        st.stop()

    risk_score = int(scored["score"])
    model_score = int(scored["model_score"])
    tier = str(scored["band"])
    tier_blurb, bg_color, text_color = _BANDS.get(tier, _BANDS["Needs review"])
    p_default = float(scored["default_probability"])
    repay_pct = (1.0 - p_default) * 100.0
    triggered_rules = list(scored["rules_applied"])
    contributions = pd.DataFrame(scored["contributions"])
    shap_vals = contributions["shap"].to_numpy(dtype=float)
    shap_base = float(scored["shap_base"])
    X_model = pd.DataFrame(
        [contributions["value"].tolist()],
        columns=contributions["feature"].tolist(),
    )

    # ──────────────────────────────────────────────────────────────────────────
    # HEADER
    # ──────────────────────────────────────────────────────────────────────────
    st.title("Will this applicant pay the loan back?")
    st.markdown(
        f"**Safety score: {risk_score}/100** — {tier_blurb} "
        f"The model estimates about **{repay_pct:.0f}%** chance of paying on time."
    )
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
                    SAFETY SCORE
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
                f"Before company rules: **{model_score}**. After rules: **{risk_score}**. "
                f"Estimated miss-payment chance: **{p_default * 100:.0f}%**."
            )
        else:
            st.caption(f"Estimated miss-payment chance: **{p_default * 100:.0f}%**.")

    with drivers_col:
        st.markdown("#### What helped or hurt?")
        st.caption("Based on the same machine-learning model — updated every time you move a slider.")

        factor_df = pd.DataFrame({
            "Feature": contributions["feature"],
            "Label": contributions["label"],
            "Impact": -shap_vals,
        })
        factor_df = factor_df.sort_values("Impact", ascending=False)

        top_positive = factor_df[factor_df["Impact"] > 0].head(4)
        top_negative = factor_df[factor_df["Impact"] < 0].head(4)

        left_f, right_f = st.columns(2)
        with left_f:
            st.markdown("**Helped the score**")
            if top_positive.empty:
                st.caption("Nothing stood out on the positive side.")
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
            st.markdown("**Hurt the score**")
            if top_negative.empty:
                st.caption("Nothing stood out on the negative side.")
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
        st.markdown("#### Company policy overrides")
        st.caption("Hard limits applied on top of the model score (like internal lending rules).")
        for rule in triggered_rules:
            st.info(rule)

    with st.expander("Detailed chart (for analysts)", expanded=False):
        st.caption(
            "Each bar is one factor. Red = pushes toward missing payments. Blue = pushes toward paying on time."
        )
        wf_fig = _waterfall_figure(X_model, shap_vals, shap_base)
        st.pyplot(wf_fig, clear_figure=True)
        plt.close(wf_fig)

    st.divider()

    with st.expander("How accurate is the model? (test data)", expanded=False):
        display_metrics = metrics_df.drop(columns=["confusion_matrix"], errors="ignore")
        numeric_cols = display_metrics.select_dtypes(include=[np.number]).columns
        formatted = display_metrics.copy()
        for c in numeric_cols:
            formatted[c] = formatted[c].map(lambda v: f"{float(v):.4f}")
        st.dataframe(formatted, hide_index=True, width="stretch")

    with st.expander("What mattered most when the model was trained?", expanded=False):
        summary_path = _ROOT / "outputs" / "shap_summary.png"
        if summary_path.is_file():
            st.caption("Overview across ~300k past applicants — not just this person.")
            st.image(str(summary_path), width="stretch")
        else:
            st.warning("Run `python src/explain.py` to generate the training summary chart.")


if __name__ == "__main__":
    main()
