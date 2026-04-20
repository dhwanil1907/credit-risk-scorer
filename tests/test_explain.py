# =============================================================================
# Tests for src/explain.py
#
# These tests verify that the SHAP explainability functions work correctly —
# that SHAP values are the right shape and that both chart types are saved
# as real image files.
#
# SHAP (SHapley Additive exPlanations) measures how much each input feature
# contributed to a specific prediction. Positive = pushed toward default,
# negative = pushed toward safe.
#
# All tests train a tiny XGBoost model inline — no real data or saved models needed.
#
# How to run: python -m pytest tests/test_explain.py -v
# =============================================================================

from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.explain import compute_shap_values, plot_shap_summary, plot_shap_waterfall


def _make_xy_and_model() -> tuple[pd.DataFrame, pd.Series, XGBClassifier]:
    """
    Create a tiny fake dataset and train a quick XGBoost model on it.

    50 rows, 10 features — small enough that TreeExplainer runs in under a second.
    Roughly 15% of rows are labelled as defaults (1) so both classes exist.
    We ensure at least 3 defaults are present so XGBoost doesn't raise an error.

    Returns the feature table (X), the labels (y), and the trained model.
    """
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((50, 10)), columns=[f"f{i}" for i in range(10)])

    # Label ~15% of rows as defaults
    y = pd.Series((rng.random(50) > 0.85).astype(np.int64))

    # Safety check: if by chance no defaults were generated, force 3 rows to be defaults
    if int(y.sum()) == 0:
        y.iloc[:3] = 1

    # Train a small, fast XGBoost model — just enough trees to have something to explain
    model = XGBClassifier(n_estimators=30, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    return X, y, model


def test_compute_shap_values_shape() -> None:
    """
    Verify that SHAP values have the same shape as the input data.

    For 50 applicants with 10 features each, we expect a 50×10 matrix of SHAP values —
    one score per feature per applicant, showing how much each feature pushed
    the prediction up or down for that specific person.
    """
    X, _y, model = _make_xy_and_model()
    shap_values, explainer = compute_shap_values(model, X)

    # SHAP matrix must match input dimensions exactly: one value per (applicant, feature) pair
    assert shap_values.shape == X.shape

    # The explainer object must also be returned (needed for waterfall plots)
    assert explainer is not None


def test_plot_shap_summary_saves_file(tmp_path: Path) -> None:
    """
    Verify that the global SHAP summary chart is saved as a real PNG file.

    The summary chart shows which features matter most across ALL applicants —
    a bird's-eye view of what drives default predictions in general.
    Features with larger bars have more influence on outcomes overall.

    tmp_path is a temporary folder that pytest creates and cleans up automatically.
    We check that the file exists and isn't empty (a blank file would indicate a silent error).
    """
    X, _y, model = _make_xy_and_model()
    shap_values, _explainer = compute_shap_values(model, X)

    out = tmp_path / "shap_summary.png"
    plot_shap_summary(shap_values, X, out, max_display=15)

    # File must exist and have content (not an empty/corrupt PNG)
    assert out.is_file(), "Summary PNG was not created"
    assert out.stat().st_size > 100, "Summary PNG file appears to be empty"


def test_plot_shap_waterfall_saves_file(tmp_path: Path) -> None:
    """
    Verify that the local SHAP waterfall chart is saved as a real PNG file.

    The waterfall chart explains ONE specific applicant's prediction:
    it shows which features pushed their score up (toward default) or
    down (toward safe), starting from the average prediction and ending
    at this person's final score.

    idx=0 means we're explaining the first applicant in the dataset.
    tmp_path is a temporary folder pytest creates and cleans up automatically.
    """
    X, _y, model = _make_xy_and_model()
    _shap_values, explainer = compute_shap_values(model, X)

    out = tmp_path / "shap_waterfall.png"
    plot_shap_waterfall(explainer, X, idx=0, save_path=out)

    # File must exist and have content (not an empty/corrupt PNG)
    assert out.is_file(), "Waterfall PNG was not created"
    assert out.stat().st_size > 100, "Waterfall PNG file appears to be empty"
