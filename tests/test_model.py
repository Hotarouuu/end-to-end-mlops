import os

from src.models.model import import_model
from dotenv import load_dotenv
import yaml
import xgboost
from src.models.helpers import load_config

load_dotenv()


config = load_config(os.getenv("CONFIG"))


def test_import_model():
    model, decode_map = import_model(config)

    assert model is not None
    assert decode_map is not None
    assert isinstance(model, xgboost.sklearn.XGBClassifier)
    assert isinstance(decode_map, dict)
