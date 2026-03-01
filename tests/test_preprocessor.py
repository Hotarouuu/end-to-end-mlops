import pytest
import numpy as np
import pandas as pd
from farm_detection.data.preprocess import Preprocessor


@pytest.fixture
def sample_df():
    data = [
        [90, 42, 43, 20.87974371, 82.00274423, 6.502985292000001, 202.9355362, "rice"],
        [85, 58, 41, 21.77046169, 80.31964408, 7.038096361, 226.6555374, "rice"],
    ]
    columns = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"]
    return pd.DataFrame(data, columns=columns)


def test_fit_transform_returns_float_columns(sample_df):
    preprocessor = Preprocessor()
    X_scaled, _ = preprocessor.fit_transform(
        sample_df.drop("label", axis=1), sample_df["label"]
    )
    assert all(dt == float for dt in X_scaled.dtypes)


def test_fit_transform_encodes_labels(sample_df):
    preprocessor = Preprocessor()
    _, y_encoded = preprocessor.fit_transform(
        sample_df.drop("label", axis=1), sample_df["label"]
    )
    assert len(y_encoded) == len(sample_df)
    assert set(y_encoded).issubset(set(range(len(sample_df["label"].unique()))))


def test_fit_transform_applies_log_to_humidity_and_rainfall(sample_df):
    preprocessor = Preprocessor()
    X_scaled, _ = preprocessor.fit_transform(
        sample_df.drop("label", axis=1).copy(), sample_df["label"]
    )
    # After log transform, humidity and rainfall columns are replaced
    assert "humidity" not in X_scaled.columns
    assert "rainfall" not in X_scaled.columns
    assert "humidity_log" in X_scaled.columns
    assert "rainfall_log" in X_scaled.columns


def test_fit_then_transform_matches_fit_transform(sample_df):
    p1 = Preprocessor()
    X_ft, y_ft = p1.fit_transform(
        sample_df.drop("label", axis=1).copy(), sample_df["label"]
    )

    p2 = Preprocessor()
    p2.fit(sample_df.drop("label", axis=1).copy(), sample_df["label"])
    X_t, y_t = p2.transform(
        sample_df.drop("label", axis=1).copy(), sample_df["label"]
    )

    np.testing.assert_array_almost_equal(X_ft.values, X_t)
    np.testing.assert_array_equal(y_ft, y_t)


def test_transform_output_dtype_is_float(sample_df):
    preprocessor = Preprocessor()
    preprocessor.fit(sample_df.drop("label", axis=1).copy(), sample_df["label"])
    X_scaled, _ = preprocessor.transform(
        sample_df.drop("label", axis=1).copy(), sample_df["label"]
    )
    assert X_scaled.dtype == float
