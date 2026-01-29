from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import pickle

# Load model & encoder ONCE
with open("soil_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

FEATURES = ["Nitrogen", "Phosphorus", "Potassium", "pH", "Moisture"]

app = FastAPI(title="Soil Quality Prediction API")


class SoilInput(BaseModel):
    Nitrogen: float = Field(..., ge=0, le=200)
    Phosphorus: float = Field(..., ge=0, le=200)
    Potassium: float = Field(..., ge=0, le=200)
    pH: float = Field(..., ge=0, le=14)
    Moisture: float = Field(..., ge=0, le=100)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: SoilInput):
    X = pd.DataFrame(
        [[data.Nitrogen, data.Phosphorus, data.Potassium, data.pH, data.Moisture]],
        columns=FEATURES
    )

    pred = model.predict(X)[0]
    proba = model.predict_proba(X).max()

    label = le.inverse_transform([pred])[0]

    return {
        "prediction": label,
        "confidence": round(proba * 100, 2)
    }
