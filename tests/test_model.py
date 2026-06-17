import os

import xgboost
import yaml
from dotenv import load_dotenv

from src.models.helpers import load_config
from src.models.model import import_model

load_dotenv()


config = load_config(os.getenv("CONFIG"))


def test_import_model():
    """Test that import_model returns a valid XGBClassifier model and decode map.

    Verifies that:
    - The model is not None
    - The decode_map is not None
    - The model is an instance of XGBClassifier
    - The decode_map is a dictionary
    """
    model, decode_map = import_model(config)

    assert model is not None
    assert decode_map is not None
    assert isinstance(model, xgboost.sklearn.XGBClassifier)
    assert isinstance(decode_map, dict)
