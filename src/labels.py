"""Plain-English names for model columns shown in the API and dashboard."""
from __future__ import annotations

FEATURE_LABELS: dict[str, str] = {
    "EXT_SOURCE_1": "Credit bureau score #1",
    "EXT_SOURCE_2": "Credit bureau score #2",
    "EXT_SOURCE_3": "Credit bureau score #3",
    "EXT_MEAN": "Average credit bureau score",
    "CREDIT_INCOME_RATIO": "Loan size vs. income",
    "ANNUITY_INCOME_RATIO": "Yearly payments vs. income",
    "DEBT_STRESS": "Overall debt pressure",
    "EXT_MEAN_X_ANNUITY_RATIO": "Credit score × payment burden",
    "EXT_MEAN_X_CREDIT_RATIO": "Credit score × loan size",
    "AGE_YEARS": "Age",
    "YEARS_EMPLOYED": "Years at current job",
    "AMT_CREDIT": "Loan amount",
    "AMT_INCOME_TOTAL": "Annual income",
    "AMT_ANNUITY": "Yearly loan payment",
    "AMT_GOODS_PRICE": "Price of what they are buying",
    "CREDIT_TERM": "Loan length (months)",
    "bureau_max_overdue": "Worst late-payment streak (days)",
    "bureau_count": "Past loans on credit file",
    "bureau_total_debt": "Total existing debt",
    "CODE_GENDER": "Gender",
    "NAME_CONTRACT_TYPE": "Loan type",
    "CNT_CHILDREN": "Number of children",
}


def feature_label(column: str) -> str:
    return FEATURE_LABELS.get(column, column.replace("_", " ").title())
