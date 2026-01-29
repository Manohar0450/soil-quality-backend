from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()

with open("soil_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

class SoilInput(BaseModel):
    Nitrogen: float
    Phosphorus: float
    Potassium: float
    pH: float
    Moisture: float

@app.post("/predict")
def predict(data: SoilInput):
    X = pd.DataFrame([[data.Nitrogen, data.Phosphorus, data.Potassium, data.pH, data.Moisture]],
                     columns=["Nitrogen","Phosphorus","Potassium","pH","Moisture"])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X).max()
    return {
        "prediction": le.inverse_transform([pred])[0],
        "confidence": round(prob * 100, 2)
    }
