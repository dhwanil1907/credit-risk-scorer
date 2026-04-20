import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import apply_guardrails


def _make_X(
    annuity_ratio: float = 0.1, credit_ratio: float = 1.0, max_overdue: float = 0.0
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ANNUITY_INCOME_RATIO": annuity_ratio,
                "CREDIT_INCOME_RATIO": credit_ratio,
                "bureau_max_overdue": max_overdue,
            }
        ]
    )


def test_no_rules_triggered_returns_model_score():
    score, rules = apply_guardrails(75, _make_X(annuity_ratio=0.1, credit_ratio=1.5, max_overdue=0))
    assert score == 75
    assert rules == []


def test_annuity_ratio_above_1_caps_at_30():
    score, rules = apply_guardrails(91, _make_X(annuity_ratio=2.67))
    assert score == 30
    assert len(rules) == 1
    assert "annual income" in rules[0].lower()


def test_annuity_ratio_above_05_caps_at_55():
    score, rules = apply_guardrails(80, _make_X(annuity_ratio=0.7))
    assert score == 55
    assert "50%" in rules[0]


def test_credit_ratio_above_10_caps_at_45():
    score, rules = apply_guardrails(85, _make_X(annuity_ratio=0.1, credit_ratio=12.0))
    assert score == 45
    assert any("×" in r for r in rules)


def test_overdue_above_60_reduces_score_by_15():
    score, rules = apply_guardrails(70, _make_X(max_overdue=90.0))
    assert score == 55
    assert any("overdue" in r.lower() for r in rules)


def test_harshest_cap_wins_when_multiple_rules_fire():
    score, rules = apply_guardrails(91, _make_X(annuity_ratio=2.67, max_overdue=90.0))
    assert score == 15
    assert len(rules) == 2


def test_score_never_goes_below_zero():
    score, rules = apply_guardrails(10, _make_X(max_overdue=90.0))
    assert score == 0
