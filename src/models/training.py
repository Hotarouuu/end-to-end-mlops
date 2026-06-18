import logging
import os
import sys

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
import yaml
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

from src.models.helpers import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


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
    model_name = "XGBOOST"
    mlflow.xgboost.autolog(
        log_models=True, log_datasets=True, registered_model_name=model_name
    )
    with mlflow.start_run() as run:

        run_id = run.info.run_id

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
        metric = cross_val_score(
            model, X, y, cv=5, scoring="neg_log_loss"
        )  # Using cv to generate the metrics, since I'm using all the data to train

        mlflow.log_metric("LogLoss", np.mean(-metric))

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

        logging.info("Model training completed. LogLoss: {}".format(np.mean(-metric)))

        logging.info(
            "Model logged to MLflow with name 'farm-detection-xgb-model'. Please check the MLflow UI for details."
        )

        logging.info("Model saved.")

        return metric, model_name, run_id


def promote_model_if_better(model_name, metric_name, benchmark, client, run_id):
    """Promote a model version to Production if it meets or exceeds the benchmark metric.

    This function:
    - Retrieves the current metric value from the specified run
    - Checks if a Production version exists for the model
    - If no Production version exists and the metric passes the benchmark, promotes the model
    - If a Production version exists, compares metrics and promotes the new version if it's better

    Args:
        model_name (str): Name of the MLflow model to promote
        metric_name (str): Name of the metric to evaluate (e.g., 'LogLoss')
        benchmark (float): Threshold value that the metric must meet or exceed
        client (mlflow.MlflowClient): MLflow client instance for model registry operations
        run_id (str): MLflow run ID containing the model metrics

    Returns:
        None
    """

    # Search for run metrics
    run = client.get_run(run_id)
    metrics = run.data.metrics
    current_metric = metrics.get(metric_name)

    if current_metric is None:
        logging.info(f"Metric '{metric_name}' not found in the run")
        return

    # Check if there is a production version of the model
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])

    if not prod_versions:
        # No production version exists, check if it passes benchmark
        latest_version = client.get_latest_versions(model_name)[0]
        logging.info(f"Current Metric: {current_metric} (benchmark: {benchmark})")

        if current_metric > benchmark:  # menor MSE = melhor
            logging.info(f"It didn't pass the benchmark. No promotion.")
            return
        else:
            client.transition_model_version_stage(
                name=model_name, version=latest_version.version, stage="Production"
            )
            logging.info(
                f"Version {latest_version.version} was promoted to Production (first time)"
            )
    else:
        # There is a production version, compare metrics
        prod_version = prod_versions[0]
        prod_run = client.get_run(prod_version.run_id)
        prod_metric = prod_run.data.metrics.get(metric_name)

        if current_metric < prod_metric:
            client.transition_model_version_stage(
                name=model_name, version=prod_version.version, stage="Archived"
            )
            latest_version = client.get_latest_versions(model_name, stages=["None"])[0]
            client.transition_model_version_stage(
                name=model_name, version=latest_version.version, stage="Production"
            )
            logging.info(
                f"Version {latest_version.version} substituted {prod_version.version} in Production"
            )
        else:
            logging.info(f"Didn't improve (prod: {prod_metric}, new: {current_metric})")


if __name__ == "__main__":

    client = mlflow.MlflowClient()
    metric, model_name, run_id = train()

    # Auto-promote model if it passes benchmark
    benchmark = 0.33  # LogLoss threshold

    promote_model_if_better(model_name, "LogLoss", benchmark, client, run_id)
