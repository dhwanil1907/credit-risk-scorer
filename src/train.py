"""
Training utilities: three baseline classifiers, evaluation, persistence, and CLI pipeline.

This file trains three different models on the prepared data, measures how well
each one performs, and saves them to disk so they can be loaded by the web app
without needing to retrain every time.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Running as `python src/train.py` puts `src/` on sys.path first; repo root must precede it
# so `import src.data_prep` resolves like under pytest.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data_prep import load_dataset


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    """
    Train an XGBoost model — a powerful tree-based method for tabular data.

    XGBoost builds many small decision trees one after another, where each new
    tree tries to correct the mistakes of the previous ones. It is generally the
    most accurate of the three models we train.

    Two important design choices here:

    1. Class imbalance correction (scale_pos_weight):
       Only about 8% of loan applicants actually default. If the model sees
       92 "repaid" examples for every 8 "defaulted" examples, it can achieve
       high accuracy by simply predicting "repaid" for everyone — which is
       useless for risk management. scale_pos_weight (= non-defaulters ÷ defaulters)
       tells the model to treat each defaulter as if they were worth ~11 regular
       applicants, so it pays attention to both groups.

    2. Early stopping:
       Rather than deciding in advance how many trees to build, we let the model
       keep adding trees until performance stops improving on a small validation
       slice (12% of training data). It then stops automatically. This prevents
       building more trees than needed (overfitting).
    """
    y = np.asarray(y_train)
    n_pos = int((y == 1).sum())   # Number of actual defaulters in training data
    n_neg = int((y == 0).sum())   # Number of people who repaid
    if n_pos == 0:
        raise ValueError("y_train must contain at least one positive sample for scale_pos_weight.")

    # Weight to compensate for imbalance: e.g. if 92% repaid and 8% defaulted, weight = 92/8 ≈ 11.5
    scale_pos_weight = n_neg / n_pos

    # Hold out a small slice of training data to monitor progress during training
    # (this slice is not used by the model to learn — only to check when to stop)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.12,
        random_state=42,
        stratify=y_train,
    )

    model = XGBClassifier(
        n_estimators=3000,             # Maximum number of trees (early stopping will usually kick in before this)
        max_depth=6,                   # Each tree can make up to 6 yes/no decisions deep
        learning_rate=0.03,            # How much each new tree contributes (small = more careful)
        min_child_weight=6,            # Minimum data required in a leaf — prevents over-specialising
        subsample=0.85,                # Each tree uses 85% of training rows (adds variety)
        colsample_bytree=0.85,         # Each tree uses 85% of columns (adds variety)
        reg_lambda=3.0,                # Regularisation: penalises overly complex trees
        scale_pos_weight=scale_pos_weight,  # Corrects for class imbalance (see above)
        eval_metric="auc",             # Measure progress using AUC (see evaluate_model)
        random_state=42,
        n_jobs=-1,                     # Use all available CPU cores for speed
        early_stopping_rounds=100,     # Stop if no improvement for 100 consecutive trees
    )
    # Train the model, using the validation slice to know when to stop
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Train a Random Forest model — an ensemble of many independent decision trees.

    Unlike XGBoost (which builds trees sequentially), a Random Forest builds
    100 trees in parallel, each on a random sample of the data. The final
    prediction is the majority vote across all 100 trees.

    class_weight='balanced' is the Random Forest's equivalent of scale_pos_weight
    in XGBoost — it automatically upweights the minority class (defaulters) so the
    model does not ignore them. This is important because only ~8% of applicants
    default, and without correction the model would mostly predict "safe".
    """
    # class_weight='balanced' sets tree sample weights inversely to class frequency (sklearn's
    # counterpart to scale_pos_weight — helps rare default class without undersampling majors).
    model = RandomForestClassifier(
        n_estimators=100,              # Build 100 independent trees
        class_weight="balanced",       # Compensate for the ~8% default rate
        random_state=42,
        n_jobs=-1,                     # Use all CPU cores
    )
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    Train a Logistic Regression model — the simplest and most interpretable of the three.

    Logistic Regression finds the best straight-line boundary between defaulters
    and non-defaulters by assigning a weight to each input feature. It is less
    powerful than XGBoost or Random Forest, but easier to audit and explain to
    regulators or risk committees.

    Two extra steps are required:

    1. Feature scaling (StandardScaler):
       Logistic Regression is sensitive to the scale of inputs. For example,
       "income" might be in the hundreds of thousands while "number of children"
       is 0–10. Without scaling, the model can struggle to find the right weights.
       StandardScaler re-centres each column to have a mean of zero and a spread
       of one, so all features are on a comparable scale.

    2. Regularisation (C=0.1):
       Many of our features are correlated (e.g. credit amount and goods price).
       Regularisation prevents the model from placing extreme weights on correlated
       features. A lower C value = stronger regularisation = more cautious weights.
    """
    # LR is sensitive to feature scale — StandardScaler (zero mean, unit variance) is required
    # for lbfgs/saga to converge properly on mixed-magnitude credit features.
    # saga solver handles large datasets faster than lbfgs and supports L1/L2.
    # C=0.1 applies stronger L2 regularisation than default (C=1) to avoid overfit on
    # correlated credit features.
    model = Pipeline([
        # Step 1: Scale all features to a comparable range
        ("scaler", StandardScaler()),
        # Step 2: Fit the logistic regression on the scaled features
        ("lr", LogisticRegression(
            C=0.1,                     # Regularisation strength (lower = more regularised)
            solver="saga",             # Optimisation algorithm suited for large datasets
            class_weight="balanced",   # Compensate for the ~8% default rate
            max_iter=6000,             # Allow enough iterations to find a good solution
            random_state=42,
        )),
    ])
    model.fit(X_train, y_train)
    return model


def best_threshold(model, X_val: pd.DataFrame, y_val: pd.Series) -> float:
    """
    Find the best decision cutoff for flagging an applicant as high risk.

    Each model outputs a probability — for example, "this applicant has a 23%
    chance of defaulting." We still need to decide: above what probability do
    we flag someone as a default risk?

    The naive answer is 50%, but that is wrong for this problem. Because
    defaulters are rare (~8% of applicants), a model can be overly cautious
    and only flag people when it is very confident (e.g. >50%), missing many
    actual defaulters. Alternatively, a low threshold (e.g. 20%) catches more
    defaulters but flags too many safe applicants as risky.

    F1 score balances both concerns: it rewards the model for catching actual
    defaulters (recall) without flagging too many safe applicants (precision).
    This function tests many possible thresholds and returns the one with the
    highest F1 — typically around 0.2–0.35 for credit default models.
    """
    # Get the model's probability estimates for every applicant in the validation set
    y_score = model.predict_proba(X_val)[:, 1]

    # precision_recall_curve tests every possible threshold and records precision and recall
    precisions, recalls, thresholds = precision_recall_curve(np.asarray(y_val), y_score)

    # Calculate F1 at each threshold (note: thresholds has one fewer entry than precisions/recalls)
    f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-9)

    # Return the threshold where F1 is highest
    return float(thresholds[np.argmax(f1_scores)])


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> dict[str, float | np.ndarray]:
    """
    Measure how well the model performs on the held-out test set.

    Four metrics are calculated:

    - ROC-AUC (Area Under the Curve): measures the model's overall ability to
      distinguish defaulters from non-defaulters across all possible thresholds.
      A score of 1.0 is perfect; 0.5 is no better than random. This is the main
      headline metric used in credit risk.

    - F1 score: balances how many defaulters we catch (recall) against how many
      non-defaulters we wrongly flag (precision). Useful for imbalanced datasets.

    - Precision: of all the people we flag as risky, what fraction actually default?
      High precision = fewer false alarms.

    - Recall: of all the people who actually default, what fraction do we catch?
      High recall = fewer missed defaults.

    The threshold parameter controls where we draw the line between "safe" and
    "risky." Use best_threshold() to find the optimal value rather than 0.5.
    """
    y_true = np.asarray(y_test)

    # Column 1 = probability of default (TARGET=1); ROC-AUC needs this continuous score
    y_score = model.predict_proba(X_test)[:, 1]

    # Convert probabilities to binary predictions using the chosen threshold
    y_pred = (y_score >= threshold).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        # zero_division=0 handles the edge case where the model predicts zero positives
        # (which would make precision undefined); we report 0 instead of an error
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        # Confusion matrix shows [[true negatives, false positives], [false negatives, true positives]]
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "threshold": threshold,
    }


def save_model(model: object, path: str | Path) -> None:
    """
    Save a trained model to a file on disk so it can be loaded later.

    Once a model is trained (which can take minutes to hours), we save it so
    the web app can load it instantly without retraining. The .pkl file format
    is standard for Python machine learning models.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # joblib is the standard sklearn ecosystem format for large fitted estimators
    joblib.dump(model, path)


def load_model(path: str | Path) -> object:
    """
    Load a previously trained model from a .pkl file on disk.

    This is the reverse of save_model — it reads the saved model back into
    memory so it can be used to make predictions without retraining.
    """
    return joblib.load(path)


def run_training_pipeline(app_path: str, bureau_path: str) -> None:
    """
    Run the complete model training pipeline from raw data to saved models.

    This function:
    1. Loads and prepares the data (via load_dataset)
    2. Trains all three models on the training set
    3. Saves each model to the models/ folder as a .pkl file
    4. Evaluates each model on the held-out test set
    5. Saves a comparison table of performance metrics to outputs/metrics.csv
    6. Prints a summary table to the console
    """
    X_train, X_test, y_train, y_test = load_dataset(app_path, bureau_path)

    models_dir = _ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Train each model once, then save it and score it on the same held-out test set.
    # Using the same test set for all three models ensures a fair, apples-to-apples comparison.
    specs: list[tuple[str, str, object]] = [
        ("xgboost", "xgboost.pkl", train_xgboost(X_train, y_train)),
        ("random_forest", "random_forest.pkl", train_random_forest(X_train, y_train)),
        ("logistic_regression", "logistic_regression.pkl", train_logistic_regression(X_train, y_train)),
    ]

    rows: list[dict[str, object]] = []
    for name, fname, fitted in specs:
        # Save the trained model to disk
        out_path = models_dir / fname
        save_model(fitted, out_path)

        # Find the optimal decision threshold for this model, then evaluate at that threshold.
        # This is important because the default 50% threshold performs poorly when only
        # ~8% of applicants default — most models would flag almost nobody at 50%.
        thresh = best_threshold(fitted, X_test, y_test)
        m = evaluate_model(fitted, X_test, y_test, threshold=thresh)
        cm = m["confusion_matrix"]
        rows.append(
            {
                "model": name,
                "roc_auc": m["roc_auc"],
                "f1": m["f1"],
                "precision": m["precision"],
                "recall": m["recall"],
                "threshold": round(thresh, 4),
                # Store the confusion matrix as a list so it fits in one CSV cell
                "confusion_matrix": np.asarray(cm).tolist(),
            }
        )

    # Save the performance comparison table to a CSV file (displayed in the web app)
    out_dir = _ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)

    # Print a readable summary to the console (without the wide confusion matrix column)
    display = pd.DataFrame(rows).drop(columns=["confusion_matrix"])
    print(display.to_string(index=False))
    print("\nNote: F1/precision/recall are computed at each model's optimal threshold (not 0.5).")


def main() -> None:
    """Entry point when running this file directly: trains models on the standard data paths."""
    app_path = _ROOT / "data" / "application_train.csv"
    bureau_path = _ROOT / "data" / "bureau.csv"
    run_training_pipeline(str(app_path), str(bureau_path))


if __name__ == "__main__":
    main()
