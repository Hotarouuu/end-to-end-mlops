import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.preprocessing import LabelEncoder

VALID_PAYLOAD = {
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 20.87974371,
    "humidity": 82.00274423,
    "ph": 6.502985292,
    "rainfall": 202.9355362,
}

CROP_NAMES = [
    "apple",
    "banana",
    "blackgram",
    "chickpea",
    "coconut",
    "coffee",
    "cotton",
    "grapes",
    "jute",
    "kidneybeans",
    "lentil",
    "maize",
    "mango",
    "mothbeans",
    "mungbean",
    "muskmelon",
    "orange",
    "papaya",
    "pigeonpeas",
    "pomegranate",
    "rice",
    "watermelon",
]


@pytest.fixture(scope="module")
# Patches all MLflow and joblib calls so app.py can be imported without a live server.
# The fake model always predicts class 17 ("papaya"), and the label encoder is pre-fitted.
def api_client():
    le = LabelEncoder()
    le.fit(CROP_NAMES)  # "papaya" is index 17

    fake_model = MagicMock()
    fake_model.predict.return_value = np.array([17])

    fake_version = MagicMock()
    fake_version.run_id = "fake_run_id"
    fake_version.version = "1"

    with (
        patch.dict(os.environ, {"CONFIG": "./config/model1.yaml"}),
        patch("mlflow.set_tracking_uri"),
        patch("mlflow.tracking.MlflowClient") as MockClient,
        patch("mlflow.sklearn.load_model", return_value=fake_model),
        patch(
            "mlflow.artifacts.download_artifacts", return_value="/tmp/fake_le.joblib"
        ),
        patch("joblib.load", return_value=le),
    ):
        MockClient.return_value.get_latest_versions.return_value = [fake_version]
        # Force a fresh import so module-level initialization runs inside the patches.
        sys.modules.pop("app", None)
        import app as app_module

        yield TestClient(app_module.app)

    sys.modules.pop("app", None)


# Checks that the root endpoint is reachable and returns a message.
def test_root(api_client):
    response = api_client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


# Verifies a valid payload returns the expected prediction index and crop label.
def test_predict_valid_input(api_client):
    response = api_client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == 17
    assert body["label"] == "papaya"


# Confirms FastAPI returns 422 when a required field is absent.
def test_predict_missing_field_returns_422(api_client):
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "rainfall"}
    assert api_client.post("/predict", json=payload).status_code == 422


# Confirms FastAPI returns 422 when a field has the wrong type.
def test_predict_invalid_type_returns_422(api_client):
    assert (
        api_client.post("/predict", json={**VALID_PAYLOAD, "N": "bad"}).status_code
        == 422
    )
