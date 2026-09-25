"""Business rule caps applied on top of the ML risk score."""
from __future__ import annotations

import pandas as pd


def apply_guardrails(model_score: int, X_row: pd.DataFrame) -> tuple[int, list[str]]:
    """
    Apply policy overrides after the model score. Harshest cap wins; overdue is a deduction.

    Returns:
        adjusted_score: final score after all rules
        triggered: human-readable descriptions of rules that fired
    """
    score = model_score
    triggered: list[str] = []

    annuity_ratio = (
        float(X_row["ANNUITY_INCOME_RATIO"].iloc[0])
        if "ANNUITY_INCOME_RATIO" in X_row.columns
        else 0.0
    )
    credit_ratio = (
        float(X_row["CREDIT_INCOME_RATIO"].iloc[0]) if "CREDIT_INCOME_RATIO" in X_row.columns else 0.0
    )
    max_overdue = (
        float(X_row["bureau_max_overdue"].iloc[0]) if "bureau_max_overdue" in X_row.columns else 0.0
    )

    if annuity_ratio > 1.0:
        if score > 30:
            score = 30
        triggered.append(
            f"Yearly loan payments are more than their annual income "
            f"({annuity_ratio:.1f}× income) — score capped at 30"
        )
    elif annuity_ratio > 0.5:
        if score > 55:
            score = 55
        triggered.append(
            f"Loan payments take over 50% of income ({annuity_ratio:.0%} of income) — score capped at 55"
        )

    if credit_ratio > 10.0:
        if score > 45:
            score = 45
        triggered.append(
            f"Loan is {credit_ratio:.1f}× annual income — too large — score capped at 45"
        )

    if max_overdue > 60:
        score = max(0, score - 15)
        triggered.append(
            f"Credit file shows {int(max_overdue)} days overdue on a past loan — score reduced by 15"
        )

    return score, triggered
