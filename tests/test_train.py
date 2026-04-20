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
    500 rows, 10 numeric features, ~8% positive class (stratified-style counts).
    Ensures both classes exist for ROC-AUC and scale_pos_weight.
    """
    rng = np.random.default_rng(42)
    n_train, n_test = 400, 100
    n_features = 10
    cols = [f"f{i}" for i in range(n_features)]
    X_train = pd.DataFrame(rng.standard_normal((n_train, n_features)), columns=cols)
    X_test = pd.DataFrame(rng.standard_normal((n_test, n_features)), columns=cols)

    pos_train = int(round(0.08 * n_train))
    pos_test = int(round(0.08 * n_test))
    y_train_arr = np.array([1] * pos_train + [0] * (n_train - pos_train), dtype=np.int64)
    y_test_arr = np.array([1] * pos_test + [0] * (n_test - pos_test), dtype=np.int64)
    rng.shuffle(y_train_arr)
    rng.shuffle(y_test_arr)
    y_train = pd.Series(y_train_arr)
    y_test = pd.Series(y_test_arr)
    return X_train, X_test, y_train, y_test


def test_train_xgboost_returns_model() -> None:
    X_train, X_test, y_train, _y_test = _make_synthetic_split()
    model = train_xgboost(X_train, y_train)
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)


def test_train_random_forest_returns_model() -> None:
    X_train, X_test, y_train, _y_test = _make_synthetic_split()
    model = train_random_forest(X_train, y_train)
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)


def test_train_logistic_regression_returns_model() -> None:
    X_train, X_test, y_train, _y_test = _make_synthetic_split()
    model = train_logistic_regression(X_train, y_train)
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)


def test_evaluate_model_returns_metrics() -> None:
    X_train, X_test, y_train, y_test = _make_synthetic_split()
    model = train_logistic_regression(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    for key in ("roc_auc", "f1", "precision", "recall", "confusion_matrix"):
        assert key in metrics

    assert 0.0 <= float(metrics["roc_auc"]) <= 1.0
