"""Integration tests for the FastAPI prediction API (``app.py``).

``app.py`` connects to MLflow and loads the model at *import time*, so a
standard ``from app import app`` at module level would fail without a live
MLflow server.  The ``api_client`` fixture below patches every external
dependency (MLflow tracking, model registry, artifact download, joblib) before
the module is imported, then yields a ``TestClient`` backed by the patched app.

Fixtures
--------
api_client
    Module-scoped ``TestClient`` with all external dependencies mocked.
    Re-imports ``app`` fresh inside the patch context so module-level
    initialization runs against the mocks.

Notes
-----
``app`` is deleted from ``sys.modules`` before and after the fixture so that
each test session always boots from a clean slate.
"""

import os
import sys
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

VALID_PAYLOAD = {
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 20.87974371,
    "humidity": 82.00274423,
    "ph": 6.502985292000001,
    "rainfall": 202.9355362,
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def api_client():
    """Yield a ``TestClient`` with all MLflow/joblib dependencies mocked.

    Steps
    -----
    1. Build a ``LabelEncoder`` pre-fitted on a small set of crop names so
       ``inverse_transform`` works without real training data.
    2. Build a fake sklearn model whose ``predict`` always returns ``[0]``
       (corresponding to the first class in the label encoder, i.e. "mango").
       A second fake model is configured to return ``[17]`` for the
       ``VALID_PAYLOAD`` expected class.
    3. Patch all MLflow calls that run at ``app.py`` module level:
       ``set_tracking_uri``, ``MlflowClient``, ``sklearn.load_model``, and
       ``artifacts.download_artifacts``.
    4. Patch ``joblib.load`` to return the fake label encoder instead of
       reading a file.
    5. Force a fresh import of ``app`` inside the patch context so the
       module-level initialisation executes against the mocks.
    6. Yield the ``TestClient`` to the tests.
    7. Remove ``app`` from ``sys.modules`` on teardown so subsequent sessions
       get a clean import.
    """
    # Build fake label encoder: class index 17 maps to "papaya"
    le = LabelEncoder()
    crop_names = [
        "apple", "banana", "blackgram", "chickpea", "coconut",
        "coffee", "cotton", "grapes", "jute", "kidneybeans",
        "lentil", "maize", "mango", "mothbeans", "mungbean",
        "muskmelon", "orange", "papaya", "pigeonpeas", "pomegranate",
        "rice", "watermelon",
    ]
    le.fit(crop_names)  # "papaya" is at index 17 in sorted order

    # Fake model: always predicts class 17
    fake_model = MagicMock()
    fake_model.predict.return_value = np.array([17])

    # Fake MLflow version metadata
    fake_version = MagicMock()
    fake_version.run_id = "fake_run_id"
    fake_version.version = "1"

    with (
        patch.dict(os.environ, {"CONFIG": "./config/model1.yaml"}),
        patch("mlflow.set_tracking_uri"),
        patch("mlflow.tracking.MlflowClient") as MockClient,
        patch("mlflow.sklearn.load_model", return_value=fake_model),
        patch(
            "mlflow.artifacts.download_artifacts",
            return_value="/tmp/fake_label_encoder.joblib",
        ),
        patch("joblib.load", return_value=le),
    ):
        MockClient.return_value.get_latest_versions.return_value = [fake_version]

        # Remove any cached module so the fresh import runs inside the patches
        sys.modules.pop("app", None)
        import app as app_module

        yield TestClient(app_module.app)

    # Cleanup: remove the patched module so it doesn't leak into other sessions
    sys.modules.pop("app", None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_root_returns_welcome_message(api_client):
    """GET / must return HTTP 200 with a 'message' key in the response body."""
    response = api_client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_predict_returns_prediction_and_label(api_client):
    """POST /predict must return HTTP 200 with 'prediction' (int) and 'label' (str) keys."""
    response = api_client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert "label" in body
    assert isinstance(body["prediction"], int)
    assert isinstance(body["label"], str)


def test_predict_known_input_returns_expected_label(api_client):
    """The canonical papaya data-point must be classified as class 17 / 'papaya'."""
    response = api_client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == 17
    assert body["label"] == "papaya"


def test_predict_missing_field_returns_422(api_client):
    """POST /predict with a missing required field must return HTTP 422."""
    incomplete_payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "rainfall"}
    response = api_client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422


def test_predict_invalid_type_returns_422(api_client):
    """POST /predict with a non-numeric value for a numeric field must return HTTP 422."""
    bad_payload = {**VALID_PAYLOAD, "N": "not-a-number"}
    response = api_client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_different_valid_input(api_client):
    """POST /predict must return HTTP 200 with 'prediction' and 'label' for any valid input."""
    payload = {
        "N": 27,
        "P": 60,
        "K": 17,
        "temperature": 23.0,
        "humidity": 63.64698302,
        "ph": 7.026795359,
        "rainfall": 64.42177127,
    }
    response = api_client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert "label" in body

