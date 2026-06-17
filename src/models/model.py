import logging

import joblib
import mlflow
from mlflow.tracking import MlflowClient


def import_model(config, is_local=False):
    """Load a trained model and its decode map from MLflow.

    Args:
        config (dict): Configuration dictionary containing artifacts settings.
        is_local (bool, optional): Whether running locally without Docker. Defaults to False.

    Returns:
        tuple: A tuple containing the loaded model and the decode map.
    """
    logging.info("Loading the model for prediction")

    if is_local:  # If running locally without Docker, set the tracking URI to localhost
        mlflow.set_tracking_uri("http://localhost:5000")
    else:
        mlflow.set_tracking_uri("http://mlflow:5000")

    client = MlflowClient()
    model_name = config["artifacts"]["model_name"]

    latest_version_info = client.get_latest_versions(model_name, stages=["Production"])
    run_id = latest_version_info[0].run_id
    latest_version = latest_version_info[0].version

    model_uri = f"models:/{model_name}/{latest_version}"

    model = mlflow.xgboost.load_model(model_uri)

    logging.info(f"Loaded version {latest_version} of '{model_name}'.")
    logging.info("Model loaded successfully")
    logging.info("Generating decode map...")

    decode_map_path = mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{run_id}/{config['artifacts']['decode_path']}"
    )
    decode_map = joblib.load(decode_map_path)

    return model, decode_map
