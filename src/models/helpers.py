import yaml


def load_config(path):
    """
    Load configuration from a YAML file.

    Args:
        path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration dictionary loaded from the YAML file.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)
