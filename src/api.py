from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal, Optional
import pandas as pd
import os
from .routes.predict import Model
from .models.models import SupervisedSessionData, UnsupervisedSessionData
from utils.gcp import CloudStorageOps
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class RequestPayload(BaseModel):
    mode: Literal["supervised", "unsupervised"]
    supervised_data: Optional[SupervisedSessionData] = None
    unsupervised_data: Optional[UnsupervisedSessionData] = None

app = FastAPI()

def save_inference_log(data: dict, prediction: int, mode: str):
    filename = "data/new_data_sup.csv" if mode == "supervised" else "data/new_data_unsup.csv"

    data_with_prediction = data.copy()
    data_with_prediction["prediction"] = prediction

    df = pd.DataFrame([data_with_prediction])

    file_exists = os.path.exists(filename)
    df.to_csv(filename, mode="a", header=not file_exists, index=False)
    gcs = CloudStorageOps("ml-anomaly-detection")
    gcs.upload_to_bucket(filename, f"src/{filename}")

@app.post("/detect_anomaly")
def detect_anomaly(payload: RequestPayload):
    try:
        if payload.mode == "supervised":
            if payload.supervised_data is None:
                raise HTTPException(status_code=400, detail="Missing data for supervised mode.")
            model = Model.load_model(supervised=True)
            input_data = payload.supervised_data.dict()
            features = Model.build_features_supervised(payload.supervised_data.dict())
            prediction = model.predict([features])
            save_inference_log(input_data, int(prediction[0]), mode="supervised")

        elif payload.mode == "unsupervised":
            if payload.unsupervised_data is None:
                raise HTTPException(status_code=400, detail="Missing data for unsupervised mode.")
            model = Model.load_model(supervised=False)
            input_data = payload.unsupervised_data.dict()
            features = Model.build_features_unsupervised(payload.unsupervised_data.dict())
            prediction = model.predict([features])
            save_inference_log(input_data, int(prediction[0]), mode="unsupervised")

        logging.info(f"Prediction: {prediction[0]}")
        return {"prediction": int(prediction[0])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))