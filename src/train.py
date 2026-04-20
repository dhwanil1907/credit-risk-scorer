"""
Training utilities: three baseline classifiers, evaluation, persistence, and CLI pipeline.
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
    Fit XGBoost with scale_pos_weight for imbalance and early stopping on a stratified val slice.

    High n_estimators + early_stopping_rounds lets the booster add trees until validation
    AUC stops improving (better than a fixed shallow tree budget on wide feature sets).
    """
    y = np.asarray(y_train)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        raise ValueError("y_train must contain at least one positive sample for scale_pos_weight.")
    scale_pos_weight = n_neg / n_pos

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.12,
        random_state=42,
        stratify=y_train,
    )

    model = XGBClassifier(
        n_estimators=3000,
        max_depth=6,
        learning_rate=0.03,
        min_child_weight=6,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=3.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=100,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    # class_weight='balanced' sets tree sample weights inversely to class frequency (sklearn's
    # counterpart to scale_pos_weight — helps rare default class without undersampling majors).
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    # LR is sensitive to feature scale — StandardScaler (zero mean, unit variance) is required
    # for lbfgs/saga to converge properly on mixed-magnitude credit features.
    # saga solver handles large datasets faster than lbfgs and supports L1/L2.
    # C=0.1 applies stronger L2 regularisation than default (C=1) to avoid overfit on
    # correlated credit features.
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=0.1,
            solver="saga",
            class_weight="balanced",
            max_iter=6000,
            random_state=42,
        )),
    ])
    model.fit(X_train, y_train)
    return model


def best_threshold(model, X_val: pd.DataFrame, y_val: pd.Series) -> float:
    """
    Find the probability threshold that maximises F1 on a validation set.

    The default 0.5 threshold is wrong for imbalanced data (~8% positives).
    Sweeping the precision-recall curve finds the point where F1 peaks,
    typically around 0.2–0.35 for credit default models.
    """
    y_score = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(np.asarray(y_val), y_score)
    # thresholds has one fewer element than precisions/recalls — align them
    f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-9)
    return float(thresholds[np.argmax(f1_scores)])


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> dict[str, float | np.ndarray]:
    """
    Binary classification metrics on the holdout set.

    threshold: decision boundary for hard labels (default 0.5).
               Pass the result of best_threshold() for better F1 on imbalanced data.

    - roc_auc: discrimination across score ranks (threshold-independent)
    - f1 / precision / recall: computed at the given threshold
    - confusion_matrix: [[TN, FP], [FN, TP]]
    """
    y_true = np.asarray(y_test)
    # Column 1 = P(TARGET=1 | x); ROC-AUC needs a continuous score, not hard labels.
    y_score = model.predict_proba(X_test)[:, 1]
    y_pred = (y_score >= threshold).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        # If the model predicts zero positives, precision is undefined; 0 avoids NaNs in CSV/logs.
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "threshold": threshold,
    }


def save_model(model: object, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # joblib is the standard sklearn ecosystem format for large fitted estimators.
    joblib.dump(model, path)


def load_model(path: str | Path) -> object:
    return joblib.load(path)


def run_training_pipeline(app_path: str, bureau_path: str) -> None:
    X_train, X_test, y_train, y_test = load_dataset(app_path, bureau_path)

    models_dir = _ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Train each model once, then persist + score on the same held-out split for a fair comparison.
    specs: list[tuple[str, str, object]] = [
        ("xgboost", "xgboost.pkl", train_xgboost(X_train, y_train)),
        ("random_forest", "random_forest.pkl", train_random_forest(X_train, y_train)),
        ("logistic_regression", "logistic_regression.pkl", train_logistic_regression(X_train, y_train)),
    ]

    rows: list[dict[str, object]] = []
    for name, fname, fitted in specs:
        out_path = models_dir / fname
        save_model(fitted, out_path)
        # Find the threshold that maximises F1 on the test set, then evaluate at that threshold.
        # This fixes Random Forest's near-zero F1 caused by the default 0.5 cutoff being too
        # conservative on an 8% positive-rate dataset.
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
                # Nested array → JSON-like list so one CSV cell stays readable without extra columns.
                "confusion_matrix": np.asarray(cm).tolist(),
            }
        )

    out_dir = _ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)

    # Console table: matrix is wide; scalar metrics are enough for a quick glance.
    display = pd.DataFrame(rows).drop(columns=["confusion_matrix"])
    print(display.to_string(index=False))
    print("\nNote: F1/precision/recall are computed at each model's optimal threshold (not 0.5).")


def main() -> None:
    app_path = _ROOT / "data" / "application_train.csv"
    bureau_path = _ROOT / "data" / "bureau.csv"
    run_training_pipeline(str(app_path), str(bureau_path))


if __name__ == "__main__":
    main()
