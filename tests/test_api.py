from fastapi.testclient import TestClient

from app import app

VALID_PAYLOAD = {
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 20.87974371,
    "humidity": 82.00274423,
    "ph": 6.502985292,
    "rainfall": 202.9355362,
}

client = TestClient(app)


def test_api_return_valid_response():

    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Farm Detection API!"}


def test_api_prediction():  # Be aware that we are testing the PREDICT ENDPOINT and NOT the model itself

    response = client.post("/predict", json=VALID_PAYLOAD)
    body = response.json()
    print(body)
    assert body["prediction"] == 1
