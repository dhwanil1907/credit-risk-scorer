from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.explain import compute_shap_values, plot_shap_summary, plot_shap_waterfall


def _make_xy_and_model() -> tuple[pd.DataFrame, pd.Series, XGBClassifier]:
    """50 rows, 10 features — small enough for fast TreeExplainer in every test."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((50, 10)), columns=[f"f{i}" for i in range(10)])
    y = pd.Series((rng.random(50) > 0.85).astype(np.int64))
    if int(y.sum()) == 0:
        y.iloc[:3] = 1
    model = XGBClassifier(n_estimators=30, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    return X, y, model


def test_compute_shap_values_shape() -> None:
    X, _y, model = _make_xy_and_model()
    shap_values, explainer = compute_shap_values(model, X)
    assert shap_values.shape == X.shape
    assert explainer is not None


def test_plot_shap_summary_saves_file(tmp_path: Path) -> None:
    X, _y, model = _make_xy_and_model()
    shap_values, _explainer = compute_shap_values(model, X)
    out = tmp_path / "shap_summary.png"
    plot_shap_summary(shap_values, X, out, max_display=15)
    assert out.is_file()
    assert out.stat().st_size > 100


def test_plot_shap_waterfall_saves_file(tmp_path: Path) -> None:
    X, _y, model = _make_xy_and_model()
    _shap_values, explainer = compute_shap_values(model, X)
    out = tmp_path / "shap_waterfall.png"
    plot_shap_waterfall(explainer, X, idx=0, save_path=out)
    assert out.is_file()
    assert out.stat().st_size > 100
