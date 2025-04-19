import logging
import os
from typing import Literal, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.ml.metrics import SystemMetrics
from utils.gcp import CloudStorageOps
from workers.pub_new_data import publish_new_data

from .models.models import SupervisedSessionData, UnsupervisedSessionData
from .routes.predict import Model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class RequestPayload(BaseModel):
    mode: Literal["supervised", "unsupervised"]
    supervised_data: Optional[SupervisedSessionData] = None
    unsupervised_data: Optional[UnsupervisedSessionData] = None


class ResponseModel(BaseModel):
    prediction: int

    class Config:
        schema_extra = {
            "example": {
                "prediction": 1,
            }
        }


app = FastAPI(
    title="Anomaly Detection API",
    description="API for detecting anomalies in data using machine learning models.",
    version="0.1.1",
    contact={"name": "Juan Vieira", "url": "https://www.linkedin.com/in/juanvieira85/"},
)


def save_inference_log(data: dict, prediction: int, mode: str):
    filename = (
        "data/new_data_sup.csv" if mode == "supervised" else "data/new_data_unsup.csv"
    )

    data_with_prediction = data.copy()
    data_with_prediction["prediction"] = prediction

    df = pd.DataFrame([data_with_prediction])

    file_exists = os.path.exists(filename)
    df.to_csv(filename, mode="a", header=not file_exists, index=False)
    gcs = CloudStorageOps("ml-anomaly-detection")
    gcs.upload_to_bucket(filename, f"src/{filename}")


@app.post(
    "/detect_anomaly",
    response_model=ResponseModel,
    summary="""
    Detect anomalies in data using both machine learning models,
    supervised and unsupervised.
    """,
)
def detect_anomaly(payload: RequestPayload):
    try:
        if payload.mode == "supervised":
            if payload.supervised_data is None:
                raise HTTPException(
                    status_code=400, detail="Missing data for supervised mode."
                )
            model, model_path = Model.load_model(supervised=True)
            input_data = payload.supervised_data.dict()
            features = Model.build_features_supervised(payload.supervised_data.dict())
            prediction = model.predict([features])
            publish_new_data(
                {
                    "data": input_data,
                    "prediction": int(prediction[0]),
                    "mode": "supervised",
                }
            )
            metrics = SystemMetrics(
                infer_function=model.predict,
                model_path=model_path,
                sample_data=[features],
            )

            local_metrics_path, metrics_filename = metrics.export_metrics(
                mode=payload.mode
            )

            gcs = CloudStorageOps("ml-anomaly-detection")
            gcs.upload_to_bucket(local_metrics_path, f"metrics/{metrics_filename}")

        elif payload.mode == "unsupervised":
            if payload.unsupervised_data is None:
                raise HTTPException(
                    status_code=400, detail="Missing data for unsupervised mode."
                )
            model, model_path = Model.load_model(supervised=False)
            input_data = payload.unsupervised_data.dict()
            features = Model.build_features_unsupervised(
                payload.unsupervised_data.dict()
            )
            prediction = model.predict([features])
            publish_new_data(
                {
                    "data": input_data,
                    "prediction": int(prediction[0]),
                    "mode": "unsupervised",
                }
            )
            metrics = SystemMetrics(
                infer_function=model.predict,
                model_path=model_path,
                sample_data=[features],
            )

            local_metrics_path, metrics_filename = metrics.export_metrics(
                mode=payload.mode
            )

            gcs = CloudStorageOps("ml-anomaly-detection")
            gcs.upload_to_bucket(local_metrics_path, f"metrics/{metrics_filename}")

        logging.info(f"Prediction: {prediction[0]}")
        return {"prediction": int(prediction[0])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
