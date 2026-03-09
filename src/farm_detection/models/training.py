from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import pandas as pd
from farm_detection.models.model import GNBWithEncoding
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
    #remote_server_uri = "http://localhost:5000" # -> Use this if running locally without Docker
    mlflow.set_tracking_uri(remote_server_uri)
    logging.info("Tracking URI set to {}".format(remote_server_uri))

    mlflow.set_experiment("Naive Bayes Experiment")
    logging.info("Experiment set to Naive Bayes Experiment")
    mlflow.sklearn.autolog(log_models=True, registered_model_name="farm-detection-gnb-model")
    with mlflow.start_run():

        logging.info("Loading data from {}".format(config["data"]["train_path"]))

        df = pd.read_csv(config["data"]["train_path"])

        logging.info("Data loaded successfully. Shape: {}".format(df.shape))

        logging.info("Splitting data into training and testing sets")

        X = df.drop(config["data"]["target"], axis=1)
        y = df[config["data"]["target"]]

        train_X, test_X, train_y, test_y = train_test_split(
            X, y, test_size=0.2, random_state=42
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

        model = GNBWithEncoding(
            priors=config["model"]["variables"]["priors"],
            var_smoothing=config["model"]["variables"]["var_smoothing"],
        )

        logging.info("Starting model training")

        model.fit(train_X, train_y)

        pred = model.predict(test_X)
        print(classification_report(test_y, pred, digits=4))

        joblib.dump(model.label_encoder, config['artifacts']['label_encoder'])

        mlflow.log_artifact(config['artifacts']['label_encoder'])


        logging.info(
            "Model training completed. Classification report:\n{}".format(
                classification_report(test_y, pred, digits=4)
            )
        )

        logging.info("Model logged to MLflow with name 'farm-detection-gnb-model' and registered model name 'farm-detection-gnb-model'")

        print("Model saved.")


if __name__ == "__main__":
    train()
