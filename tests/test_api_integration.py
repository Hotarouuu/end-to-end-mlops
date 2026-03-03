import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

VALID_PAYLOAD = {
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 20.87974371,
    "humidity": 82.00274423,
    "ph": 6.502985292000001,
    "rainfall": 202.9355362,
}


def test_root_returns_welcome_message():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_predict_returns_prediction_and_label():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert "label" in body
    assert isinstance(body["prediction"], int)
    assert isinstance(body["label"], str)


def test_predict_known_input_returns_expected_label():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == 17
    assert body["label"] == "papaya"


def test_predict_missing_field_returns_422():
    incomplete_payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "rainfall"}
    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422


def test_predict_invalid_type_returns_422():
    bad_payload = {**VALID_PAYLOAD, "N": "not-a-number"}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_different_valid_input():
    payload = {
        "N": 27,
        "P": 60,
        "K": 17,
        "temperature": 23.0,
        "humidity": 63.64698302,
        "ph": 7.026795359,
        "rainfall": 64.42177127,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert "label" in body
