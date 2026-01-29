from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
import os

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

with open(os.path.join(BASE_DIR, "soil_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)

FEATURES = ["Nitrogen", "Phosphorus", "Potassium", "pH", "Moisture"]

class SoilInput(BaseModel):
    Nitrogen: float
    Phosphorus: float
    Potassium: float
    pH: float
    Moisture: float

@app.get("/")
def health():
    return {"message": "Soil Quality API is running 🚀"}

@app.post("/")
def predict(data: SoilInput):
    X = pd.DataFrame([[data.Nitrogen, data.Phosphorus, data.Potassium, data.pH, data.Moisture]],
                     columns=FEATURES)

    pred = model.predict(X)[0]
    confidence = model.predict_proba(X).max()

    label = le.inverse_transform([pred])[0]

    return {
        "prediction": label,
        "confidence": round(confidence * 100, 2)
    }
