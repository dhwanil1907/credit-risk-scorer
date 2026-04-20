"""
SHAP explainability for the trained XGBoost credit model: global summary + local waterfall.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Repo root on sys.path so `python src/explain.py` can import `src.*` (see train.py).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Non-interactive backend first — SHAP/matplotlib must not open a GUI in CI or SSH sessions.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import shap

from src.data_prep import load_dataset


def _coerce_shap_matrix(raw: object, expected_shape: tuple[int, int]) -> np.ndarray:
    """
    TreeExplainer.shap_values sometimes returns a list [values_class0, values_class1] for binary
    targets; we keep contributions toward class 1 (default / positive label), matching XGBoost
    binary:objective convention used in training.
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


def compute_shap_values(model, X: pd.DataFrame) -> tuple[np.ndarray, shap.TreeExplainer]:
    """
    Build a TreeExplainer for the fitted XGBoost estimator and compute SHAP values for every row.

    Returns SHAP matrix aligned with X (same shape) plus the explainer for waterfall plots
    (needs expected_value and the same tree structure).
    """
    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X)
    matrix = _coerce_shap_matrix(raw, (X.shape[0], X.shape[1]))
    return matrix, explainer


def plot_shap_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    save_path: str | Path,
    max_display: int = 15,
) -> None:
    """
    Beeswarm summary: features sorted by mean |SHAP| (global impact). max_display caps how many
    rows appear — keeps the figure readable when many columns exist.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(shap_values, X, show=False, max_display=max_display)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close("all")


def _expected_value_scalar(explainer: shap.TreeExplainer) -> float:
    """Pick the scalar base value that matches the SHAP vector (binary margin output)."""
    ev = explainer.expected_value
    ev_arr = np.asarray(ev).ravel()
    if ev_arr.size > 1:
        # Binary margin: match positive-class SHAP row length / convention.
        return float(ev_arr[1])
    return float(ev_arr[0])


def plot_shap_waterfall(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
    idx: int,
    save_path: str | Path,
) -> None:
    """
    Local explanation for X.iloc[idx]: each bar is that feature's SHAP contribution vs the
    model baseline (expected_value) for this row's prediction path through the trees.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    row_df = X.iloc[[idx]]
    raw = explainer.shap_values(row_df)
    vals = _coerce_shap_matrix(raw, (1, X.shape[1]))[0]
    base = _expected_value_scalar(explainer)
    explanation = shap.Explanation(
        values=vals,
        base_values=base,
        data=X.iloc[idx].to_numpy(),
        feature_names=list(X.columns),
    )
    shap.plots.waterfall(explanation, show=False)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close("all")


def run_explain_pipeline(
    model_path: str,
    app_path: str,
    bureau_path: str,
    sample_size: int = 1000,
) -> None:
    """
    End-to-end: reload the saved XGBoost artifact, explain a random holdout slice, write plots.

    We subsample X_test for speed (SHAP on full test + deep trees is expensive); random_state
    keeps the slice reproducible across runs.
    """
    model = joblib.load(model_path)
    _X_train, X_test, _y_train, _y_test = load_dataset(app_path, bureau_path)
    n = min(sample_size, len(X_test))
    X_sample = X_test.sample(n=n, random_state=42)

    shap_values, explainer = compute_shap_values(model, X_sample)
    out_dir = _ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "shap_summary.png"
    waterfall_path = out_dir / "shap_waterfall.png"

    plot_shap_summary(shap_values, X_sample, summary_path, max_display=15)
    plot_shap_waterfall(explainer, X_sample, idx=0, save_path=waterfall_path)

    # Quick sanity check: which features dominate mean |SHAP| on this sample (no plot parsing).
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(-mean_abs)
    ranked = [X_sample.columns[i] for i in order[:15]]
    print("Top 15 features by mean |SHAP| on sample:", ", ".join(ranked))


def main() -> None:
    run_explain_pipeline(
        str(_ROOT / "models" / "xgboost.pkl"),
        str(_ROOT / "data" / "application_train.csv"),
        str(_ROOT / "data" / "bureau.csv"),
    )


if __name__ == "__main__":
    main()
