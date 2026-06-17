"""Pytest configuration for the farm-detection test suite.

Inserts the project root directory into ``sys.path`` so that top-level modules
(e.g. ``app.py``) are importable from within the ``tests/`` directory without
requiring an editable install.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import joblib
import mlflow
import numpy as np
import pytest
import xgboost

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Creating fixture for session scope of the mlflow


@pytest.fixture(scope="session", autouse=True)
def mock_mlflow_session():
    """Pytest fixture that mocks MLflow and model components for testing.

    This fixture sets up mock objects for MLflow tracking, model loading,
    artifact downloading, and the MlflowClient. It patches relevant modules
    to allow tests to run without actual MLflow interactions.

    Yields:
        dict: A dictionary containing:
            - 'model': Mock XGBClassifier instance
            - 'decode_map': Mock decode map dictionary
    """
    mock_model_obj = MagicMock()
    mock_model_obj.__class__ = xgboost.sklearn.XGBClassifier
    mock_model_obj.predict.return_value = np.array([0])

    mock_version_info = MagicMock()
    mock_version_info.run_id = "test-run-id"
    mock_version_info.version = "1"

    mock_decode_path = "/tmp/mock_decode_map.pkl"
    mock_decode = {0: "apple", 1: "banana"}
    joblib.dump(mock_decode, mock_decode_path)

    with (
        patch("mlflow.set_tracking_uri"),
        patch("mlflow.xgboost.load_model", return_value=mock_model_obj),
        patch("mlflow.artifacts.download_artifacts", return_value=mock_decode_path),
        patch("src.models.model.MlflowClient") as mock_client_class,
        patch("app.model") as mock_model_obj,
        patch("app.decode_map") as mock_decode,
    ):
        mock_client_class.return_value.get_latest_versions.return_value = [
            mock_version_info
        ]

        yield {
            "model": mock_model_obj,
            "decode_map": mock_decode,
        }


# @pytest.fixture(scope="module")
# def trained_model():
#    df = pd.read_csv("./data/Crop_recommendation.csv")
#    X, y = df.drop("label", axis=1), df["label"].astype('category').cat.codes
#    model = xgb.XGBClassifier(**config["model"]["variables"])
#    model.fit(X.copy(), y)
#    return model, set(y.unique())
