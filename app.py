import logging
import os
import sys
from contextlib import asynccontextmanager
from time import time

import joblib
import mlflow
from mlflow import shap
import pandas as pd
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

from src.models.helpers import load_config
from src.models.model import import_model

model = None
decode_map = None

# logging.Formatter.converter = time.localtime
load_dotenv()

config = load_config(os.getenv("CONFIG"))


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Loading the model before the API to avoid loading it everytime the API is requested


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):  # We are using lifespan here to load the model before taking requests and not when the code is loading. It's good for the tests as well
    global model, decode_map
    model, decode_map = import_model(config, is_local=False)
    yield


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


app = FastAPI(lifespan=lifespan)

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
    prediction = model.predict(input_df)
    label = decode_map[prediction[0]]
    logging.info(
        "Prediction made successfully. Prediction: {}, Label: {}".format(
            prediction, label
        )
    )

    return {"prediction": int(prediction[0]), "label": str(label)}


@app.post("/shap_explanation")
def shap_explanation(data: User):
    """SHAP explanation endpoint of the API.

    Args:
        data (User): Input data for SHAP explanation.

    Returns:
        dict: SHAP values for the input data.
    """
    logging.info("Received SHAP explanation request with data: {}".format(data))
    input_df = pd.DataFrame([data.model_dump()])
    explainer = shap.TreeExplainer(model, input_df)
    shap_values = explainer(input_df)
    logging.info("SHAP values calculated successfully")
    return {"shap_values": shap_values.values.tolist()}
