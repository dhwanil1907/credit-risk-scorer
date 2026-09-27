"""Tests for the scoring API: valid scores, missing fields, and out-of-range values."""
import pytest
from fastapi.testclient import TestClient

from api import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client

VALID = {
    "yearly_income": 180000,
    "loan_amount": 250000,
    "yearly_payment": 15000,
    "purchase_price": 250000,
    "age": 35,
    "years_employed": 5,
    "bureau_score_1": 0.5,
    "bureau_score_2": 0.5,
    "bureau_score_3": 0.5,
}


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_applicant(client: TestClient) -> None:
    response = client.post("/predict", json=VALID)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["score"] <= 100
    assert body["band"] in {"Looks good", "Needs review", "High risk"}
    assert body["rules_applied"] == []
    assert len(body["top_reasons"]) == 3
    assert len(body["contributions"]) >= 3
    assert "shap_base" in body
    assert body["top_reasons"][0]["label"]
    assert body["top_reasons"][0]["direction"] in {"helps", "hurts"}


def test_predict_applies_payment_cap(client: TestClient) -> None:
    payload = {**VALID, "yearly_income": 10000, "yearly_payment": 30000}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["score"] <= 30
    assert body["rules_applied"]


def test_predict_missing_fields(client: TestClient) -> None:
    response = client.post("/predict", json={"age": 35})
    assert response.status_code == 422
    fields = {err["field"] for err in response.json()["detail"]}
    assert "yearly_income" in fields
    assert "loan_amount" in fields


def test_predict_out_of_range(client: TestClient) -> None:
    response = client.post("/predict", json={**VALID, "age": 12, "bureau_score_1": 1.4})
    assert response.status_code == 422
    fields = {err["field"] for err in response.json()["detail"]}
    assert "age" in fields
    assert "bureau_score_1" in fields


def test_predict_malformed_payload(client: TestClient) -> None:
    response = client.post("/predict", json={**VALID, "yearly_income": "a lot"})
    assert response.status_code == 422
    assert response.json()["detail"][0]["field"] == "yearly_income"
