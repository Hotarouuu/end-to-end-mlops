import logging
import os
import sys

import joblib
import mlflow
import pandas as pd
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

from src.farm_detection.data.preprocess import Preprocessor

load_dotenv()


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


config = load_config(os.getenv("CONFIG"))

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Loading the model before the API to avoid loading it everytime the API is requested

logging.info("Loading the model and preprocessor for prediction")

mlflow.set_tracking_uri("http://mlflow:5000")
client = MlflowClient()
model_name = config["artifacts"]["model_name"]

latest_version_info = client.get_latest_versions(model_name, stages=["None"])
run_id = latest_version_info[0].run_id
latest_version = latest_version_info[0].version

model_uri = f"models:/{model_name}/{latest_version}"

# Why are we using label encoder and preprocessor separated if the Model Class already has them?
# For it's because the MLFlow's autologging doesn't log the preprocessor and label encoder as part of the model, so we need to load them separately
# to ensure that we have all the necessary components for making predictions in the API.

model = mlflow.sklearn.load_model(model_uri)

label_encoder_path = mlflow.artifacts.download_artifacts(
    artifact_uri=f"runs:/{run_id}/{config['artifacts']['label_encoder_name']}"
)

preprocessor = Preprocessor()
label_encoder = joblib.load(label_encoder_path)

print(f"Success! Loaded version {latest_version} of '{model_name}'.")

logging.info("Model and preprocessor loaded successfully")
logging.info("Starting the FastAPI application")

# The range of variables is based on the min-max range of each variable in the training data.
# This was selected because it represents the data the model saw during training.


class User(BaseModel):
    N: int = Field(ge=0, le=140)
    P: int = Field(ge=5, le=145)
    K: int = Field(ge=5, le=205)
    temperature: float = Field(ge=8.8, le=43.6)
    humidity: float = Field(ge=14.25, le=99.98)
    ph: float = Field(ge=0.0, le=14.0)
    rainfall: float = Field(ge=20.21, le=298.56)


app = FastAPI()

logging.info("FastAPI application started successfully")
logging.info("Defining the /predict endpoint")


@app.get("/")
def read_root():
    """Root endpoint of the API.

    Returns:
        dict: A welcome message.
    """
    logging.info("Received request at root endpoint")
    return {"message": "Welcome to the Farm Detection API!"}


@app.post("/predict")
def predict(data: User):
    """Predict endpoint of the API.

    Args:
        data (User): Input data for prediction.

    Returns:
        dict: Prediction result and corresponding label.
    """
    logging.info("Received prediction request with data: {}".format(data))
    input_data = [list(data.model_dump().values())]
    input_df = pd.DataFrame([data.model_dump()])
    logging.info("Input data for prediction: {}".format(input_data))
    input_df = preprocessor.log_transform(input_df)
    prediction = model.predict(input_df)
    label = label_encoder.inverse_transform(prediction)
    logging.info(
        "Prediction made successfully. Prediction: {}, Label: {}".format(
            prediction, label
        )
    )

    return {"prediction": int(prediction[0]), "label": str(label[0])}
