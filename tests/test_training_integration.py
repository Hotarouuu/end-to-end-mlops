import pytest
import joblib
from unittest.mock import patch, MagicMock
from farm_detection.models.training import train

MOCK_CONFIG = {
    "data": {
        "train_path": "./data/Crop_recommendation.csv",
        "features": ["N", "P", "K", "temperature", "ph", "humidity", "rainfall"],
        "target": "label",
    },
    "model": {"variables": {"priors": None, "var_smoothing": 1e-9}},
    "artifacts": {},  # filled per test via tmp_path
}


@pytest.fixture(autouse=True)
def mock_mlflow():
    """Patch MLflow to avoid requiring a running MLflow server."""
    with patch("farm_detection.models.training.mlflow") as mock:
        mock.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock.start_run.return_value.__exit__ = MagicMock(return_value=False)
        yield mock


def _config(tmp_path):
    return {
        **MOCK_CONFIG,
        "artifacts": {
            "model_path": str(tmp_path / "gaussiannb.joblib"),
            "preprocessor_path": str(tmp_path / "preprocessor.joblib"),
        },
    }


def test_train_completes_without_error(tmp_path):
    with patch(
        "farm_detection.models.training.load_config", return_value=_config(tmp_path)
    ):
        train()  # should not raise


def test_train_saves_model_and_preprocessor_to_disk(tmp_path):
    cfg = _config(tmp_path)
    with patch("farm_detection.models.training.load_config", return_value=cfg):
        train()

    assert (tmp_path / "gaussiannb.joblib").exists(), "model file not saved"
    assert (tmp_path / "preprocessor.joblib").exists(), "preprocessor file not saved"


def test_train_saved_model_can_predict(tmp_path):
    cfg = _config(tmp_path)
    with patch("farm_detection.models.training.load_config", return_value=cfg):
        train()

    model = joblib.load(cfg["artifacts"]["model_path"])
    preprocessor = joblib.load(cfg["artifacts"]["preprocessor_path"])

    sample = [[90, 6.5, 82.0, 20.8, 43, 42, 202.9]]  # features order matches training
    X_scaled = preprocessor["scaler"].transform(sample)
    prediction = model.predict(X_scaled)

    assert len(prediction) == 1
    assert isinstance(int(prediction[0]), int)


def test_train_saved_preprocessor_has_required_keys(tmp_path):
    cfg = _config(tmp_path)
    with patch("farm_detection.models.training.load_config", return_value=cfg):
        train()

    preprocessor = joblib.load(cfg["artifacts"]["preprocessor_path"])
    assert "scaler" in preprocessor
    assert "labelencoder" in preprocessor


def test_train_mlflow_run_is_started(tmp_path, mock_mlflow):
    with patch(
        "farm_detection.models.training.load_config", return_value=_config(tmp_path)
    ):
        train()

    mock_mlflow.start_run.assert_called_once()
    mock_mlflow.set_experiment.assert_called_once_with("Naive Bayes Experiment")
