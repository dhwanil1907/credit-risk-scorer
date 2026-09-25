"""
SHAP explainability for the trained XGBoost credit model: global summary + local waterfall.

This file answers the question: "Why did the model give this applicant that score?"

SHAP (SHapley Additive exPlanations) is a technique that breaks down each model
prediction into contributions from each input feature. For example, it can tell you:
"The applicant's high credit utilization pushed their risk score up by +12 points,
but their long employment history pulled it back down by -8 points."

Two types of chart are produced:
- A global summary chart: shows which features matter most across all applicants
- A waterfall chart: shows the exact breakdown for one specific applicant
"""
from __future__ import annotations

import sys
from pathlib import Path

# Repo root on sys.path so `python src/explain.py` can import `src.*` (see train.py).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Use a non-interactive chart backend — required so SHAP/matplotlib can save images
# to files without trying to open a window (important in server/automated environments)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import shap

from src.data_prep import load_dataset
from src.shap_utils import coerce_shap_matrix, expected_value_scalar


def compute_shap_values(model, X: pd.DataFrame) -> tuple[np.ndarray, shap.TreeExplainer]:
    """
    Calculate SHAP values for every applicant in the dataset.

    A SHAP value tells you how much each feature pushed this particular
    applicant's predicted default probability up or down compared to the
    average applicant. Positive SHAP value = pushed toward default.
    Negative SHAP value = pushed toward safe/repaid.

    Returns two things:
    1. A matrix of SHAP values — one row per applicant, one column per feature
    2. The explainer object itself (needed later for individual applicant charts)
    """
    # Build the SHAP explainer from the trained model's internal tree structure
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values for every row (applicant) in X
    raw = explainer.shap_values(X)

    # Normalise the output format regardless of how the library returned it
    matrix = coerce_shap_matrix(raw, (X.shape[0], X.shape[1]))
    return matrix, explainer


def plot_shap_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    save_path: str | Path,
    max_display: int = 15,
) -> None:
    """
    Create and save a global feature importance chart (beeswarm plot).

    This chart shows which features have the biggest impact on the model's
    predictions across all applicants in the sample. Features are ranked from
    most to least important, and each dot represents one applicant.

    - Features at the top matter most on average
    - Dots on the right (positive SHAP) push predictions toward default
    - Dots on the left (negative SHAP) push predictions toward safe
    - Colour indicates the feature value: red = high value, blue = low value

    max_display limits how many features are shown (top 15 by default) to
    keep the chart readable.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate the beeswarm chart (show=False means don't pop open a window)
    shap.summary_plot(shap_values, X, show=False, max_display=max_display)

    # Save the chart as a PNG image file
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close("all")


def plot_shap_waterfall(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
    idx: int,
    save_path: str | Path,
) -> None:
    """
    Create and save a personalised SHAP explanation chart for one specific applicant.

    A waterfall chart shows how the model arrived at its prediction for a single
    applicant, step by step. Starting from the average predicted probability
    (the baseline), each bar shows how much one feature pushed the score up or
    down. The final bar shows where the score ended up.

    For example:
    - Baseline (average applicant): 8% default probability
    - High credit utilization: +5% (pushed toward default)
    - Long employment history: -3% (pushed toward safe)
    - Low external credit score: +7% (pushed toward default)
    - Final predicted probability: 17%

    idx is the row number of the applicant in the dataset (0 = first applicant).
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract just the one applicant row we want to explain
    row_df = X.iloc[[idx]]

    # Calculate SHAP values specifically for this one applicant
    raw = explainer.shap_values(row_df)
    vals = coerce_shap_matrix(raw, (1, X.shape[1]))[0]

    # Get the baseline (average) prediction to start the waterfall from
    base = expected_value_scalar(explainer)

    # Package everything into the format SHAP's waterfall chart expects
    explanation = shap.Explanation(
        values=vals,                          # The SHAP contributions per feature
        base_values=base,                     # The baseline prediction to start from
        data=X.iloc[idx].to_numpy(),          # The actual feature values for this applicant
        feature_names=list(X.columns),        # Human-readable column names for the chart
    )

    # Draw the waterfall chart (show=False = save to file, don't open a window)
    shap.plots.waterfall(explanation, show=False)

    # Save the chart as a PNG image file
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close("all")


def run_explain_pipeline(
    model_path: str,
    app_path: str,
    bureau_path: str,
    sample_size: int = 1000,
) -> None:
    """
    Run the complete explanation pipeline end-to-end.

    This function:
    1. Loads the saved XGBoost model from disk
    2. Prepares the test data using the same steps as training
    3. Takes a random sample of up to 1,000 test applicants
       (computing SHAP values for all 50,000+ applicants would take too long)
    4. Calculates SHAP values for each applicant in the sample
    5. Saves the global summary chart to outputs/shap_summary.png
    6. Saves a waterfall chart for the first applicant to outputs/shap_waterfall.png
    7. Prints the top 15 most influential features to the console

    random_state=42 ensures the same sample is selected each time the script
    is run, making results reproducible and comparable across runs.
    """
    # Load the trained model from its saved file
    model = joblib.load(model_path)

    # Prepare the data (we only use the test set — it was not seen during training)
    _X_train, X_test, _y_train, _y_test = load_dataset(app_path, bureau_path)

    # Take a sample for speed — SHAP on deep trees can be slow for large datasets
    n = min(sample_size, len(X_test))
    X_sample = X_test.sample(n=n, random_state=42)

    # Calculate SHAP values for every applicant in the sample
    shap_values, explainer = compute_shap_values(model, X_sample)

    # Set up the output directory
    out_dir = _ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "shap_summary.png"
    waterfall_path = out_dir / "shap_waterfall.png"

    # Save the global feature importance chart
    plot_shap_summary(shap_values, X_sample, summary_path, max_display=15)

    # Save a personalised chart for the first applicant in the sample (index 0)
    plot_shap_waterfall(explainer, X_sample, idx=0, save_path=waterfall_path)

    # Print a quick ranking of the most influential features to the console
    # (average absolute SHAP value across all sample applicants = global importance)
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(-mean_abs)   # Sort descending (most important first)
    ranked = [X_sample.columns[i] for i in order[:15]]
    print("Top 15 features by mean |SHAP| on sample:", ", ".join(ranked))


def main() -> None:
    """Entry point when running this file directly: explains the saved XGBoost model."""
    run_explain_pipeline(
        str(_ROOT / "models" / "xgboost.pkl"),
        str(_ROOT / "data" / "application_train.csv"),
        str(_ROOT / "data" / "bureau.csv"),
    )


if __name__ == "__main__":
    main()
