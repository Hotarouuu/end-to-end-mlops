import logging
import sys

import joblib
import mlflow
import pandas as pd
import yaml
from sklearn.model_selection import cross_val_score
import numpy as np
import shap
import os
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


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


def train():
    """
    Train an XGBoost model for farm detection.

    This function:
    - Loads configuration from a YAML file
    - Sets up MLflow tracking and experiment
    - Loads training data from CSV
    - Trains a XGBoost model
    - Evaluates the model and logs results to MLflow

    Returns:
        None
    """
    config = load_config("./config/model1.yaml")

    # Setting up MLflow tracking URI and experiment

    logging.info("Setting up MLflow tracking URI and experiment")
    remote_server_uri = "http://mlflow:5000"
    # remote_server_uri = "http://localhost:5000" # -> Use this if running locally without Docker
    mlflow.set_tracking_uri(remote_server_uri)
    logging.info("Tracking URI set to {}".format(remote_server_uri))

    mlflow.set_experiment("Crop Recommendation Experiment")
    logging.info("Experiment set to XGBoost Experiment")
    mlflow.xgboost.autolog(
        log_models=True, log_datasets=True, registered_model_name="XGBOOST"
    )
    with mlflow.start_run():

        logging.info("Loading data from {}".format(config["data"]["train_path"]))

        df = pd.read_csv(config["data"]["train_path"])

        logging.info("Data loaded successfully. Shape: {}".format(df.shape))

        X = df.drop(config["data"]["target"], axis=1)
        y = df[config["data"]["target"]].astype("category").cat.codes

        logging.info("Generating decode map...")

        decode_map = dict(enumerate(df["label"].astype("category").cat.categories))

        logging.info(
            "Data split completed. Training set shape: {}, Testing set shape: {}".format(
                X.shape, y.shape
            )
        )
        logging.info(
            "Initializing the XGBoost model with parameters: {}".format(
                config["model"]["variables"]
            )
        )

        model = XGBClassifier(**config["model"]["variables"])

        logging.info("Starting model training")

        model.fit(X, y)
        logging.info("Model training completed")

        mlflow.xgboost.autolog(disable=True)
        cv = cross_val_score(
            model, X, y, cv=5, scoring="neg_log_loss"
        )  # Using cv to generate the metrics, since I'm using all the data to train

        mlflow.log_metric("LogLoss", np.mean(-cv))

        # Using SHAP for better feature importance interpretation

        logging.info("Calculating SHAP values for feature importance interpretation")

        explainer = shap.TreeExplainer(model, X)
        shap_values = explainer(
            X
        )  # using explainer here isn't the most effective way to calculate the SHAP values, but since we only have 2200 samples it's ok

        artifact_dir = os.path.abspath("artifacts")
        os.makedirs(artifact_dir, exist_ok=True)
        shap_summary_path = os.path.join(artifact_dir, "shap_summary.png")
        shap.summary_plot(
            shap_values,
            X,
            plot_type="bar",
            plot_size=(20, 15),
            class_names=decode_map,
            show=False,
        )

        plt.savefig(shap_summary_path, dpi=300, bbox_inches="tight")
        plt.close()

        joblib.dump(decode_map, os.path.join(artifact_dir, "decode_map.pkl"))

        mlflow.log_artifact(shap_summary_path, artifact_path="SHAP_FEATURE_IMPORTANCE")
        mlflow.log_artifact(
            os.path.join(artifact_dir, "decode_map.pkl"), artifact_path="DECODE_MAP"
        )

        logging.info("SHAP feature importance plot saved and logged to MLflow")
        logging.info("Decode map saved and logged to MLflow")

        logging.info("Model training completed. LogLoss: {}".format(np.mean(-cv)))

        logging.info(
            "Model logged to MLflow with name 'farm-detection-xgb-model'. Please check the MLflow UI for details."
        )

        print("Model saved.")


if __name__ == "__main__":
    train()
