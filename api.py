"""FastAPI scoring service: one applicant in, score, band, rules, and SHAP reasons out."""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.labels import feature_label
from src.score import Applicant, Scorer


class ApplicantRequest(BaseModel):
    yearly_income: float = Field(gt=0, description="Annual income in dollars.")
    loan_amount: float = Field(gt=0, description="Requested loan amount in dollars.")
    yearly_payment: float = Field(ge=0, description="Yearly payment on this loan.")
    purchase_price: float = Field(ge=0, description="Price of the goods or property.")
    age: int = Field(ge=18, le=100)
    years_employed: int = Field(ge=0, le=60)
    bureau_score_1: float = Field(ge=0, le=1)
    bureau_score_2: float = Field(ge=0, le=1)
    bureau_score_3: float = Field(ge=0, le=1)
    children: int = Field(default=0, ge=0, le=20)
    owns_car: bool = True
    owns_home: bool = True
    gender: Literal["M", "F"] = "M"
    loan_type: Literal["cash", "revolving"] = "cash"
    past_loans: int = Field(default=3, ge=0, le=100)
    worst_late_days: int = Field(default=0, ge=0, le=3650)


class ReasonResponse(BaseModel):
    feature: str
    label: str
    direction: Literal["helps", "hurts"]
    impact: float


class ContributionResponse(BaseModel):
    feature: str
    label: str
    value: float
    shap: float


class PredictResponse(BaseModel):
    score: int
    model_score: int
    band: str
    default_probability: float
    rules_applied: list[str]
    top_reasons: list[ReasonResponse]
    shap_base: float
    contributions: list[ContributionResponse]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.scorer = Scorer()
    yield


app = FastAPI(title="Credit Risk Scorer", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = []
    for err in exc.errors():
        parts = [str(part) for part in err["loc"] if part != "body"]
        errors.append({"field": ".".join(parts) or "body", "message": err["msg"]})
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(body: ApplicantRequest, request: Request) -> PredictResponse:
    result = request.app.state.scorer.score(
        Applicant(
            yearly_income=body.yearly_income,
            loan_amount=body.loan_amount,
            yearly_payment=body.yearly_payment,
            purchase_price=body.purchase_price,
            age=body.age,
            years_employed=body.years_employed,
            bureau_score_1=body.bureau_score_1,
            bureau_score_2=body.bureau_score_2,
            bureau_score_3=body.bureau_score_3,
            children=body.children,
            owns_car=body.owns_car,
            owns_home=body.owns_home,
            gender=body.gender,
            loan_type=body.loan_type,
            past_loans=body.past_loans,
            worst_late_days=body.worst_late_days,
        )
    )
    columns = list(result.features.columns)
    values = result.features.iloc[0].to_numpy()
    return PredictResponse(
        score=result.score,
        model_score=result.model_score,
        band=result.band,
        default_probability=round(result.default_probability, 6),
        rules_applied=result.rules_applied,
        top_reasons=[
            ReasonResponse(
                feature=reason.feature,
                label=reason.label,
                direction=reason.direction,
                impact=reason.impact,
            )
            for reason in result.top_reasons
        ],
        shap_base=result.shap_base,
        contributions=[
            ContributionResponse(
                feature=column,
                label=feature_label(column),
                value=float(values[index]),
                shap=float(result.shap_values[index]),
            )
            for index, column in enumerate(columns)
        ],
    )
