"""Shared helpers for normalising SHAP outputs across app.py and explain.py."""
from __future__ import annotations

import numpy as np
import shap


def coerce_shap_matrix(raw: object, expected_shape: tuple[int, int]) -> np.ndarray:
    """
    Normalise SHAP output to a matrix: rows = applicants, cols = features, class = default (1).
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


def expected_value_scalar(explainer: shap.TreeExplainer) -> float:
    """Baseline default probability for binary models (expected_value for class 1)."""
    ev_arr = np.asarray(explainer.expected_value).ravel()
    if ev_arr.size > 1:
        return float(ev_arr[1])
    return float(ev_arr[0])
