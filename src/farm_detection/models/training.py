from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import pandas as pd
from farm_detection.models.model import GNB
from farm_detection.data.preprocess import Preprocessor
import joblib
import yaml
import mlflow
import logging
import sys

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train():

    config = load_config("./config/model1.yaml")

    # Enable autologging

    logging.info("Setting up MLflow tracking URI and experiment")
    remote_server_uri = "http://mlflow:5000"
    mlflow.set_tracking_uri(remote_server_uri)
    logging.info("Tracking URI set to {}".format(remote_server_uri))

    mlflow.set_experiment("Naive Bayes Experiment")
    logging.info("Experiment set to Naive Bayes Experiment")
    mlflow.sklearn.autolog()
    with mlflow.start_run():

        logging.info("Loading data from {}".format(config["data"]["train_path"]))

        df = pd.read_csv(config["data"]["train_path"])

        logging.info("Data loaded successfully. Shape: {}".format(df.shape))

        processing = Preprocessor()

        logging.info("Starting data preprocessing")

        X_scaled, y_encoded = processing.fit_transform(
            df[config["data"]["features"]], df[config["data"]["target"]]
        )

        logging.info("Data preprocessing completed successfully")

        print(X_scaled)

        logging.info("Splitting data into training and testing sets")

        train_X, test_X, train_y, test_y = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42
        )

        logging.info(
            "Data split completed. Training set shape: {}, Testing set shape: {}".format(
                train_X.shape, test_X.shape
            )
        )
        logging.info(
            "Initializing the Naive Bayes model with priors: {} and var_smoothing: {}".format(
                config["model"]["variables"]["priors"],
                config["model"]["variables"]["var_smoothing"],
            )
        )

        model = GNB(
            priors=config["model"]["variables"]["priors"],
            var_smoothing=config["model"]["variables"]["var_smoothing"],
        )

        logging.info("Starting model training")

        model.fit(train_X, train_y)

        pred = model.predict(test_X)
        print(classification_report(test_y, pred, digits=4))

        logging.info(
            "Model training completed. Classification report:\n{}".format(
                classification_report(test_y, pred, digits=4)
            )
        )

        preprocessor = {
            "scaler": processing.scaler,
            "labelencoder": processing.label_encoder,
        }

        logging.info("Saving model and preprocessor to disk")

        joblib.dump(preprocessor, config["artifacts"]["preprocessor_path"])
        joblib.dump(model, config["artifacts"]["model_path"])

        logging.info(
            "Model and preprocessor saved successfully. Model path: {}, Preprocessor path: {}".format(
                config["artifacts"]["model_path"],
                config["artifacts"]["preprocessor_path"],
            )
        )

        mlflow.log_artifact("config/model1.yaml")
        mlflow.log_artifact(config["artifacts"]["preprocessor_path"])

        logging.info(
            "Artifacts logged to MLflow: config/model1.yaml and {}".format(
                config["artifacts"]["preprocessor_path"]
            )
        )

        print("Model saved.")


if __name__ == "__main__":
    train()
