"""Pytest configuration for the farm-detection test suite.

Inserts the project root directory into ``sys.path`` so that top-level modules
(e.g. ``app.py``) are importable from within the ``tests/`` directory without
requiring an editable install.
"""

import os
import sys
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Creating fixture for session scope of the mlflow

import pytest
from unittest.mock import patch, MagicMock
import mlflow
import joblib
import pandas as pd
import xgboost as xgb

@pytest.fixture(scope="session", autouse=True)
def mock_mlflow_session():
    """Mock do MLflow para toda a sessão de testes."""
    with patch('mlflow.set_tracking_uri'), \
         patch('mlflow.xgboost.load_model') as mock_load_model, \
         patch('mlflow.artifacts.download_artifacts'), \
         patch('mlflow.tracking.MlflowClient') as mock_client:
        
        # Mock do client.get_latest_versions
        mock_version_info = MagicMock()
        mock_version_info.run_id = "test-run-id"
        mock_version_info.version = "1"
        mock_client.return_value.get_latest_versions.return_value = [mock_version_info]
        
        # Mock do load_model
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        
        # Mock do download_artifacts
        mock_decode_path = "/tmp/mock_decode_map.pkl"
        mock_decode_map = {0: "apple", "1": "banana"}
        joblib.dump(mock_decode_map, mock_decode_path)
        mock_artifacts = MagicMock()
        mock_artifacts.download_artifacts.return_value = mock_decode_path
        
        yield {
            'model': mock_model,
            'decode_map': mock_decode_map,
            'client': mock_client
        }

@pytest.fixture(scope="module")
def trained_model():
    df = pd.read_csv("./data/Crop_recommendation.csv")
    X, y = df.drop("label", axis=1), df["label"]
    model = xgb.XGBClassifier(**config["model"]["variables"])
    model.fit(X.copy(), y)
    return model, set(y.unique())


