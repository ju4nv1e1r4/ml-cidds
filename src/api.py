from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal, Optional
from .routes.predict import Model
from .models.models import SupervisedSessionData, UnsupervisedSessionData

class RequestPayload(BaseModel):
    mode: Literal["supervised", "unsupervised"]
    supervised_data: Optional[SupervisedSessionData] = None
    unsupervised_data: Optional[UnsupervisedSessionData] = None

app = FastAPI()

@app.post("/detect_anomaly")
async def detect_anomaly(payload: RequestPayload):
    try:
        if payload.mode == "supervised":
            if payload.supervised_data is None:
                raise HTTPException(status_code=400, detail="Missing data for supervised mode.")
            model = Model.load_model(supervised=True)
            features = Model.build_features_supervised(payload.supervised_data.dict())
            prediction = model.predict([features])

        elif payload.mode == "unsupervised":
            if payload.unsupervised_data is None:
                raise HTTPException(status_code=400, detail="Missing data for unsupervised mode.")
            model = Model.load_model(supervised=False)
            features = Model.build_features_unsupervised(payload.unsupervised_data.dict())
            prediction = model.predict([features])

        return {"prediction": int(prediction[0])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))