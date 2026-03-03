import pytest
from farm_detection.models.predict import Predictor


@pytest.fixture(scope="module")
def predictor():
    return Predictor(
        model_path="./model/gaussiannb.joblib",
        preprocessor_path="./model/preprocessor.joblib",
    )


def test_predict_returns_expected_class_and_label(predictor):
    test_data = [[90, 42, 43, 20.87974371, 82.00274423, 6.502985292000001, 202.9355362]]
    class_pred, decoded_pred = predictor.predict(test_data)
    assert class_pred == [17]
    assert decoded_pred == ["papaya"]


def test_predict_returns_list_types(predictor):
    test_data = [[90, 42, 43, 20.87974371, 82.00274423, 6.502985292000001, 202.9355362]]
    class_pred, decoded_pred = predictor.predict(test_data)
    assert len(class_pred) == 1
    assert len(decoded_pred) == 1
    assert isinstance(decoded_pred[0], str)


def test_predict_raises_on_invalid_input(predictor):
    with pytest.raises(Exception):
        predictor.predict([[]])
