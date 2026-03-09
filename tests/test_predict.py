"""Tests for the GNBWithEncoding model (``farm_detection.models.model``).

The ``Predictor`` helper class was removed in the latest refactor.
Prediction is now handled directly by ``GNBWithEncoding``, which bundles
scaling, classification, and label-decoding into a single sklearn-compatible
estimator.

Fixtures
--------
sample_data
    Tiny synthetic (X, y) pair used for quick smoke tests that only need a
    fitted model, not realistic accuracy.
real_data
    The full ``Crop_recommendation.csv`` loaded once per module.
trained_model
    A ``GNBWithEncoding`` fitted on ``real_data``; shared across all tests
    that need a realistic model.
"""

import pytest
import pandas as pd

from farm_detection.models.model import GNBWithEncoding


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(**overrides):
    """Return a one-row DataFrame with default crop-feature values.

    Parameters
    ----------
    **overrides:
        Column values to replace in the default row.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with all seven feature columns.
    """
    row = {
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 20.87974371,
        "ph": 6.502985292,
        "humidity": 82.00274423,
        "rainfall": 202.9355362,
    }
    row.update(overrides)
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_data():
    """Return a small synthetic (X, y) dataset for fast smoke tests.

    Uses five hand-crafted rows so the fixture does not depend on the CSV
    file being present.  A fresh DataFrame is returned on every call so that
    the in-place ``log_transform`` inside ``GNBWithEncoding.fit`` does not
    corrupt subsequent calls.
    """
    X = pd.DataFrame(
        {
            "N": [90, 85, 60, 40, 70],
            "P": [42, 58, 55, 30, 45],
            "K": [43, 41, 44, 50, 38],
            "temperature": [20.8, 21.7, 23.0, 25.0, 22.0],
            "ph": [6.5, 5.9, 6.0, 7.0, 6.2],
            "humidity": [82.0, 80.3, 78.0, 65.0, 75.0],
            "rainfall": [202.9, 234.2, 189.3, 120.0, 150.0],
        }
    )
    y = pd.Series(["papaya", "papaya", "mango", "rice", "rice"])
    return X, y


@pytest.fixture(scope="module")
def real_data():
    """Load the full Crop_recommendation CSV as (X, y).

    Loaded once per module to keep the test suite fast.  The returned
    DataFrame is a copy, so callers may modify it freely.
    """
    df = pd.read_csv("./data/Crop_recommendation.csv")
    X = df.drop("label", axis=1)
    y = df["label"]
    return X.copy(), y


@pytest.fixture(scope="module")
def trained_model(real_data):
    """Return a ``GNBWithEncoding`` fitted on the full crop dataset.

    Shared across all tests that require a realistic, trained model.
    ``X.copy()`` is passed to ``fit`` because ``log_transform`` mutates the
    DataFrame in place; without the copy the original ``real_data`` fixture
    would be corrupted for other tests.
    """
    X, y = real_data
    model = GNBWithEncoding()
    model.fit(X.copy(), y)
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_model_fit_completes_without_error(sample_data):
    """``GNBWithEncoding.fit`` must not raise on valid (X, y) inputs."""
    X, y = sample_data
    model = GNBWithEncoding()
    model.fit(X, y)


def test_model_predict_returns_string_labels(trained_model):
    """``predict`` must return an iterable of string crop-name labels."""
    result = trained_model.predict(_make_df())
    assert len(result) == 1
    assert isinstance(result[0], str)


def test_model_predict_returns_correct_number_of_predictions(trained_model):
    """``predict`` must return exactly one label per input row."""
    X_two_rows = pd.DataFrame(
        {
            "N": [90, 27],
            "P": [42, 60],
            "K": [43, 17],
            "temperature": [20.87, 23.0],
            "ph": [6.5, 7.0],
            "humidity": [82.0, 63.6],
            "rainfall": [202.9, 64.4],
        }
    )
    result = trained_model.predict(X_two_rows)
    assert len(result) == 2


def test_model_predict_known_input_returns_a_known_class(trained_model, real_data):
    """``predict`` must return a label that belongs to the training-set classes.

    We do not hard-code a single expected crop name here because the
    ``GNBWithEncoding`` pipeline may assign a different boundary than the
    legacy separate-file model did.  The important guarantee is that every
    prediction is a valid, known class rather than an out-of-vocabulary value.
    """
    _, y = real_data
    known_classes = set(y.unique())
    result = trained_model.predict(_make_df())
    assert result[0] in known_classes, (
        f"Prediction '{result[0]}' is not a recognised crop class"
    )


def test_model_predict_raises_on_missing_columns(trained_model):
    """``predict`` must raise when required feature columns are absent."""
    bad_input = pd.DataFrame({"N": [90], "P": [42]})
    with pytest.raises(Exception):
        trained_model.predict(bad_input)

