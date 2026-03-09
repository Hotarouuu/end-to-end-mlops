"""Integration tests for the training pipeline (``farm_detection.models.training``).

These tests exercise the ``train()`` function end-to-end while mocking all
MLflow network calls so no running MLflow server is required.

The latest training refactor changed the artifact contract:

* The model itself is no longer saved to a local ``.joblib`` file; MLflow
  autolog registers it directly in the model registry.
* Only the ``LabelEncoder`` is persisted locally, at the path given by
  ``config['artifacts']['label_encoder']``.
* ``train()`` reads its configuration exclusively from
  ``load_config("./config/model1.yaml")``; tests patch that call to inject
  a temp-path config so nothing is written outside the test sandbox.

Fixtures
--------
mock_mlflow
    Auto-used fixture that replaces the ``mlflow`` module inside the
    training script with a ``MagicMock`` so network calls are no-ops.
"""

import pytest
import joblib
from sklearn.preprocessing import LabelEncoder
from unittest.mock import patch, MagicMock

from farm_detection.models.training import train


# ---------------------------------------------------------------------------
# Base configuration (matches config/model1.yaml structure)
# ---------------------------------------------------------------------------

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
        "label_encoder": "",          # overridden per test via tmp_path
        "label_encoder_name": "gnb_model_label_encoder.joblib",
    },
}


def _config(tmp_path):
    """Return a copy of ``MOCK_CONFIG`` with ``label_encoder`` pointed at *tmp_path*.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory unique to each test.

    Returns
    -------
    dict
        Config dict safe to pass to ``train()`` without touching the repo.
    """
    return {
        **MOCK_CONFIG,
        "artifacts": {
            **MOCK_CONFIG["artifacts"],
            "label_encoder": str(tmp_path / "gnb_model_label_encoder.joblib"),
        },
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_mlflow():
    """Patch MLflow inside the training module to avoid requiring a live server.

    Replaces the entire ``mlflow`` object referenced by
    ``farm_detection.models.training`` with a ``MagicMock``.  The context
    manager protocol on ``start_run`` is wired up so the ``with`` statement
    in ``train()`` executes normally.

    Yields
    ------
    MagicMock
        The patched mlflow object; tests that need to make assertions on
        MLflow calls receive it via the ``mock_mlflow`` parameter.
    """
    with patch("farm_detection.models.training.mlflow") as mock:
        mock.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock.start_run.return_value.__exit__ = MagicMock(return_value=False)
        yield mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_train_completes_without_error(tmp_path):
    """``train()`` must complete without raising any exception."""
    with patch(
        "farm_detection.models.training.load_config", return_value=_config(tmp_path)
    ):
        train()


def test_train_saves_label_encoder_to_disk(tmp_path):
    """``train()`` must persist the label encoder at the configured artifact path."""
    cfg = _config(tmp_path)
    with patch("farm_detection.models.training.load_config", return_value=cfg):
        train()

    assert (tmp_path / "gnb_model_label_encoder.joblib").exists(), (
        "label encoder file not found after training"
    )


def test_train_saved_label_encoder_is_valid(tmp_path):
    """The persisted label encoder must be a fitted ``LabelEncoder`` with crop classes."""
    cfg = _config(tmp_path)
    with patch("farm_detection.models.training.load_config", return_value=cfg):
        train()

    le = joblib.load(cfg["artifacts"]["label_encoder"])
    assert isinstance(le, LabelEncoder), "artifact is not a LabelEncoder"
    assert hasattr(le, "classes_"), "label encoder has not been fitted"
    assert len(le.classes_) > 0, "label encoder has no classes"
    assert "papaya" in le.classes_, "'papaya' missing from encoder classes"


def test_train_mlflow_experiment_is_configured(tmp_path, mock_mlflow):
    """``train()`` must set the MLflow experiment to 'Naive Bayes Experiment'."""
    with patch(
        "farm_detection.models.training.load_config", return_value=_config(tmp_path)
    ):
        train()

    mock_mlflow.set_experiment.assert_called_once_with("Naive Bayes Experiment")


def test_train_mlflow_run_is_started(tmp_path, mock_mlflow):
    """``train()`` must open exactly one MLflow run via ``mlflow.start_run``."""
    with patch(
        "farm_detection.models.training.load_config", return_value=_config(tmp_path)
    ):
        train()

    mock_mlflow.start_run.assert_called_once()


def test_train_mlflow_logs_label_encoder_artifact(tmp_path, mock_mlflow):
    """``train()`` must log the label encoder file as an MLflow artifact."""
    cfg = _config(tmp_path)
    with patch("farm_detection.models.training.load_config", return_value=cfg):
        train()

    mock_mlflow.log_artifact.assert_called_once_with(cfg["artifacts"]["label_encoder"])

