import yaml
from fastapi import FastAPI
from pydantic import BaseModel
from src.farm_detection.models.predict import Predictor
import logging
import sys
import os
from dotenv import load_dotenv
load_dotenv()

config_path = os.getenv("CONFIG")

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

config = load_config(config_path)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)


# Loading the model before the API to avoid loading it everytime the API is requested

logging.info("Loading the model and preprocessor for prediction")

model = Predictor(
    model_path=config["artifacts"]["model_path"],
    preprocessor_path=config["artifacts"]["preprocessor_path"],
)

logging.info("Model and preprocessor loaded successfully")
logging.info("Starting the FastAPI application")


class User(BaseModel):
    N: int
    P: int
    K: int
    temperature: float
    humidity: float
    ph: float
    rainfall: float


app = FastAPI()

logging.info("FastAPI application started successfully")
logging.info("Defining the /predict endpoint")

@app.get("/")
def read_root():
    logging.info("Received request at root endpoint")
    return {"message": "Welcome to the Farm Detection API!"}

@app.post("/predict")
def predict(data: User):
    logging.info("Received prediction request with data: {}".format(data))
    input_data = [list(data.model_dump().values())]
    logging.info("Input data for prediction: {}".format(input_data))
    prediction, label = model.predict(input_data)
    logging.info(
        "Prediction made successfully. Prediction: {}, Label: {}".format(
            prediction, label
        )
    )

    return {"prediction": int(prediction[0]), "label": str(label[0])}
