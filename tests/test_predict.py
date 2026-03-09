import pandas as pd
import pytest

from farm_detection.models.model import GNBWithEncoding

SAMPLE_ROW = {
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 20.87974371,
    "ph": 6.502985292,
    "humidity": 82.00274423,
    "rainfall": 202.9355362,
}


@pytest.fixture(scope="module")
# Loads the full dataset, fits a model once, and returns it with the set of known crop classes.
def trained_model():
    df = pd.read_csv("./data/Crop_recommendation.csv")
    X, y = df.drop("label", axis=1), df["label"]
    model = GNBWithEncoding()
    model.fit(X.copy(), y)
    return model, set(y.unique())


# Verifies fit() runs without errors on minimal valid data.
def test_fit_does_not_raise():
    X = pd.DataFrame(
        {
            "N": [90, 85],
            "P": [42, 58],
            "K": [43, 41],
            "temperature": [20.8, 21.7],
            "ph": [6.5, 5.9],
            "humidity": [82.0, 80.3],
            "rainfall": [202.9, 234.2],
        }
    )
    y = pd.Series(["papaya", "papaya"])
    GNBWithEncoding().fit(X, y)


# Checks that a single prediction is a string belonging to the training classes.
def test_predict_returns_known_class(trained_model):
    model, classes = trained_model
    result = model.predict(pd.DataFrame([SAMPLE_ROW]))
    assert len(result) == 1
    assert isinstance(result[0], str)
    assert result[0] in classes


# Ensures predict() returns exactly one label per input row.
def test_predict_count_matches_rows(trained_model):
    model, _ = trained_model
    assert len(model.predict(pd.DataFrame([SAMPLE_ROW, SAMPLE_ROW]))) == 2


# Confirms predict() raises when required feature columns are missing.
def test_predict_raises_on_missing_columns(trained_model):
    model, _ = trained_model
    with pytest.raises(Exception):
        model.predict(pd.DataFrame({"N": [90], "P": [42]}))
