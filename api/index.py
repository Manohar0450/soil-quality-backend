from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import pickle

# ---------------- Load model & encoder ONCE ----------------
with open("soil_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

FEATURES = ["Nitrogen", "Phosphorus", "Potassium", "pH", "Moisture"]

app = FastAPI(title="Soil Quality Prediction API")


# ---------------- Request Schema (LOWERCASE) ----------------
class SoilInput(BaseModel):
    nitrogen: float = Field(..., ge=0, le=200)
    phosphorus: float = Field(..., ge=0, le=200)
    potassium: float = Field(..., ge=0, le=200)
    ph: float = Field(..., ge=0, le=14)
    moisture: float = Field(..., ge=0, le=100)


# ---------------- Health Check ----------------
@app.get("/api/health")
def health():
    return {"status": "ok"}


# ---------------- Prediction Endpoint ----------------
@app.post("/api/predict")
def predict(data: SoilInput):

    # Convert to DataFrame with TRAINING feature names
    X = pd.DataFrame(
        [[
            data.nitrogen,
            data.phosphorus,
            data.potassium,
            data.ph,
            data.moisture
        ]],
        columns=FEATURES
    )

    # Prediction
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    label = le.inverse_transform([pred])[0]

    return {
        "soil_quality": label,
        "confidence": round(float(proba.max()), 4),
        "probabilities": {
            le.classes_[i]: round(float(p), 4)
            for i, p in enumerate(proba)
        }
    }
