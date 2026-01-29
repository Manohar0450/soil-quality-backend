from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

app = FastAPI(title="Soil Quality Prediction API")

# ---------------- Load model & encoder ----------------
with open("soil_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

FEATURES = ["Nitrogen", "Phosphorus", "Potassium", "pH", "Moisture"]

# ---------------- Request Model ----------------
class SoilInput(BaseModel):
    nitrogen: float = Field(..., ge=0, le=200)
    phosphorus: float = Field(..., ge=0, le=200)
    potassium: float = Field(..., ge=0, le=200)
    ph: float = Field(..., ge=0, le=14)
    moisture: float = Field(..., ge=0, le=100)

# ---------------- Health ----------------
@app.get("/health")
def health():
    return {"status": "ok"}

# ---------------- Predict ----------------
@app.post("/predict")
def predict(data: SoilInput):
    X = pd.DataFrame([[
        data.nitrogen,
        data.phosphorus,
        data.potassium,
        data.ph,
        data.moisture
    ]], columns=FEATURES)

    pred = model.predict(X)[0]
    probs = model.predict_proba(X)[0]

    return {
        "soil_quality": le.inverse_transform([pred])[0],
        "confidence": float(max(probs)),
        "probabilities": {
            le.classes_[i]: float(probs[i]) for i in range(len(probs))
        }
    }

# ---------------- Metrics ----------------
@app.get("/metrics")
def metrics():
    df = pd.read_csv("soil_data.csv")

    X = df[FEATURES]
    y = le.transform(df["Soil_Quality"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    y_pred = model.predict(X_test)

    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 3),
        "precision": round(precision_score(y_test, y_pred, average="weighted"), 3),
        "recall": round(recall_score(y_test, y_pred, average="weighted"), 3),
        "f1_score": round(f1_score(y_test, y_pred, average="weighted"), 3),
        "labels": le.classes_.tolist(),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }
