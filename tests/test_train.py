# =============================================================================
# Tests for src/train.py
#
# These tests verify that each model trains without errors and produces
# sensible outputs. They use a small synthetic dataset (500 rows) so they
# run quickly without needing the real Kaggle data files.
#
# How to run: python -m pytest tests/test_train.py -v
# =============================================================================

import numpy as np
import pandas as pd

from src.train import (
    evaluate_model,
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
)


def _make_synthetic_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Create a small fake dataset that mimics the real credit data.

    500 rows total (400 for training, 100 for testing), 10 numeric columns,
    with roughly 8% of rows labelled as defaults (1) — matching the real dataset's
    class imbalance. Both classes must be present so models and metrics don't crash.

    We use a fixed random seed (42) so tests produce the same result every run.
    """
    rng = np.random.default_rng(42)
    n_train, n_test = 400, 100
    n_features = 10
    cols = [f"f{i}" for i in range(n_features)]  # simple column names: f0, f1, ..., f9

    # Random numbers as feature values — the model just needs something numeric to train on
    X_train = pd.DataFrame(rng.standard_normal((n_train, n_features)), columns=cols)
    X_test = pd.DataFrame(rng.standard_normal((n_test, n_features)), columns=cols)

    # Set ~8% of rows as defaults (1), the rest as repaid (0) — mirrors real data imbalance
    pos_train = int(round(0.08 * n_train))  # 32 defaults in training
    pos_test = int(round(0.08 * n_test))    # 8 defaults in test
    y_train_arr = np.array([1] * pos_train + [0] * (n_train - pos_train), dtype=np.int64)
    y_test_arr = np.array([1] * pos_test + [0] * (n_test - pos_test), dtype=np.int64)

    # Shuffle so defaults aren't all at the top
    rng.shuffle(y_train_arr)
    rng.shuffle(y_test_arr)

    y_train = pd.Series(y_train_arr)
    y_test = pd.Series(y_test_arr)
    return X_train, X_test, y_train, y_test


def test_train_xgboost_returns_model() -> None:
    """
    Check that XGBoost trains successfully and produces one prediction per applicant.

    XGBoost is our primary model — a powerful tree-based algorithm that handles
    the 8% default rate by upweighting rare defaults during training.
    We verify it returns exactly as many predictions as there are test rows.
    """
    X_train, X_test, y_train, _y_test = _make_synthetic_split()
    model = train_xgboost(X_train, y_train)
    preds = model.predict(X_test)

    # One prediction per applicant in the test set — no more, no less
    assert len(preds) == len(X_test)


def test_train_random_forest_returns_model() -> None:
    """
    Check that Random Forest trains successfully and produces one prediction per applicant.

    Random Forest is an ensemble of many decision trees that vote on the outcome.
    Using class_weight='balanced' ensures it pays enough attention to the rare defaults.
    """
    X_train, X_test, y_train, _y_test = _make_synthetic_split()
    model = train_random_forest(X_train, y_train)
    preds = model.predict(X_test)

    # One prediction per applicant in the test set
    assert len(preds) == len(X_test)


def test_train_logistic_regression_returns_model() -> None:
    """
    Check that Logistic Regression trains successfully and produces one prediction per applicant.

    Logistic Regression is the simplest of the three models — it learns a weighted
    formula that combines all features into a single default probability.
    It requires features to be on similar scales (handled internally by StandardScaler).
    """
    X_train, X_test, y_train, _y_test = _make_synthetic_split()
    model = train_logistic_regression(X_train, y_train)
    preds = model.predict(X_test)

    # One prediction per applicant in the test set
    assert len(preds) == len(X_test)


def test_evaluate_model_returns_metrics() -> None:
    """
    Check that the evaluation function returns all required performance metrics
    and that ROC-AUC is a valid number between 0 and 1.

    What each metric means:
    - roc_auc: overall ability to rank defaulters above non-defaulters (1.0 = perfect, 0.5 = random)
    - f1: balance between catching defaults and avoiding false alarms
    - precision: of the applicants flagged as risky, how many actually defaulted?
    - recall: of all actual defaulters, how many did the model catch?
    - confusion_matrix: a 2x2 table of correct/incorrect predictions

    We use Logistic Regression here because it trains fastest on small data.
    """
    X_train, X_test, y_train, y_test = _make_synthetic_split()
    model = train_logistic_regression(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    # All five metric keys must be present in the returned dictionary
    for key in ("roc_auc", "f1", "precision", "recall", "confusion_matrix"):
        assert key in metrics

    # ROC-AUC must be a valid probability between 0 and 1
    # (even on random data it stays in this range — below 0.5 just means worse than random)
    assert 0.0 <= float(metrics["roc_auc"]) <= 1.0
