from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import pickle
import os

app = FastAPI()

# Load model files (Vercel-safe paths)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "..", "soil_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "..", "label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)

FEATURES = ["Nitrogen", "Phosphorus", "Potassium", "pH", "Moisture"]

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
def predict(payload: SoilInput):
    X = pd.DataFrame([[payload.Nitrogen, payload.Phosphorus,
                       payload.Potassium, payload.pH,
                       payload.Moisture]], columns=FEATURES)

    pred = model.predict(X)[0]
    proba = float(model.predict_proba(X).max())

    label = le.inverse_transform([pred])[0]

    return {
        "prediction": label,
        "confidence": round(proba * 100, 2)
    }
