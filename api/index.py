from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import os

app = FastAPI()

# Load model files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(BASE_DIR, "soil_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)


class SoilInput(BaseModel):
    Nitrogen: float
    Phosphorus: float
    Potassium: float
    pH: float
    Moisture: float


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/predict")
def predict(data: SoilInput):
    X = [[
        data.Nitrogen,
        data.Phosphorus,
        data.Potassium,
        data.pH,
        data.Moisture
    ]]
    pred = model.predict(X)[0]
    label = label_encoder.inverse_transform([pred])[0]
    return {"prediction": label}
