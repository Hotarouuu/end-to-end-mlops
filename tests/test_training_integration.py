import pytest
import joblib
from sklearn.preprocessing import LabelEncoder
from unittest.mock import patch, MagicMock

from farm_detection.models.training import train


MOCK_CONFIG = {
    "data": {
        "train_path": "./data/Crop_recommendation.csv",
        "features": ["N", "P", "K", "temperature", "ph", "humidity", "rainfall"],
        "target": "label",
    },
    "model": {"variables": {"priors": None, "var_smoothing": 1e-9}},
    "artifacts": {
        "model_name": "farm-detection-gnb-model",
        "model_script": "./src/farm_detection/models/model.py",
        "label_encoder": "",
        "label_encoder_name": "gnb_model_label_encoder.joblib",
    },
}


# Overrides the label_encoder path so nothing is written outside the test sandbox.
def make_config(tmp_path):
    return {**MOCK_CONFIG, "artifacts": {**MOCK_CONFIG["artifacts"], "label_encoder": str(tmp_path / "le.joblib")}}


@pytest.fixture(autouse=True)
# Replaces the mlflow module in training.py with a MagicMock so no live server is needed.
def mock_mlflow():
    with patch("farm_detection.models.training.mlflow") as mock:
        mock.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock.start_run.return_value.__exit__ = MagicMock(return_value=False)
        yield mock


# Smoke test: train() should complete without raising any exception.
def test_train_completes(tmp_path):
    with patch("farm_detection.models.training.load_config", return_value=make_config(tmp_path)):
        train()


# Verifies train() saves a fitted LabelEncoder with the expected crop classes to disk.
def test_train_saves_label_encoder(tmp_path):
    cfg = make_config(tmp_path)
    with patch("farm_detection.models.training.load_config", return_value=cfg):
        train()
    le = joblib.load(cfg["artifacts"]["label_encoder"])
    assert isinstance(le, LabelEncoder)
    assert "papaya" in le.classes_


# Checks that train() sets the correct MLflow experiment, opens a run, and logs the label encoder artifact.
def test_train_mlflow_calls(tmp_path, mock_mlflow):
    cfg = make_config(tmp_path)
    with patch("farm_detection.models.training.load_config", return_value=cfg):
        train()
    mock_mlflow.set_experiment.assert_called_once_with("Naive Bayes Experiment")
    mock_mlflow.start_run.assert_called_once()
    mock_mlflow.log_artifact.assert_called_once_with(cfg["artifacts"]["label_encoder"])

