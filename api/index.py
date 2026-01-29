from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import pickle
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ---------------- Load model & encoder ----------------
with open("soil_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Load evaluation dataset (same used during training)
df = pd.read_csv("soil_data.csv")

X = df[["Nitrogen", "Phosphorus", "Potassium", "pH", "Moisture"]]
y_true = le.transform(df["Soil_Quality"])

y_pred = model.predict(X)

# ---------------- FastAPI ----------------
app = FastAPI(title="Soil Quality Prediction API")

class SoilInput(BaseModel):
    nitrogen: float = Field(..., ge=0, le=200)
    phosphorus: float = Field(..., ge=0, le=200)
    potassium: float = Field(..., ge=0, le=200)
    ph: float = Field(..., ge=0, le=14)
    moisture: float = Field(..., ge=0, le=100)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: SoilInput):
    X_input = pd.DataFrame([[
        data.nitrogen,
        data.phosphorus,
        data.potassium,
        data.ph,
        data.moisture
    ]], columns=X.columns)

    pred = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0]

    return {
        "soil_quality": le.inverse_transform([pred])[0],
        "confidence": round(max(proba), 2),
        "probabilities": dict(zip(le.classes_, proba.round(2)))
    }

@app.get("/metrics")
def metrics():
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "precision": round(precision_score(y_true, y_pred, average="weighted"), 3),
        "recall": round(recall_score(y_true, y_pred, average="weighted"), 3),
        "f1_score": round(f1_score(y_true, y_pred, average="weighted"), 3),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "labels": le.classes_.tolist()
    }
