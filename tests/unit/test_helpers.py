from src.models.helpers import load_config
import os
from dotenv import load_dotenv
import yaml
import pytest

load_dotenv()


@pytest.fixture
def import_yaml():
    """_summary_

    Returns:
        _type_: _description_
    """
    with open(os.getenv("CONFIG"), "r") as f:
        return yaml.safe_load(f)


def test_load_config():
    """_summary_"""
    config = load_config(os.getenv("CONFIG"))
    assert isinstance(config, dict)
    assert config is not None


def test_main_keys(import_yaml):
    """_summary_

    Args:
        import_yaml (_type_): _description_
    """
    main_keys = {"data", "model", "artifacts"}
    yaml_keys = set(import_yaml.keys())

    assert main_keys.issubset(yaml_keys)


def test_secundary_keys(import_yaml):
    """_summary_

    Args:
        import_yaml (_type_): _description_
    """
    data_keys = {"train_path", "features", "target"}
    model_keys = {"type", "variables"}
    artifacts_keys = {"model_name", "model_script", "decode_path"}

    assert data_keys.issubset(set(import_yaml["data"].keys()))
    assert model_keys.issubset(set(import_yaml["model"].keys()))
    assert artifacts_keys.issubset(set(import_yaml["artifacts"].keys()))
