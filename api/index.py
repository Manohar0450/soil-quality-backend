from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

app = FastAPI(title="Soil Quality Prediction API")

# Load data
df = pd.read_csv("soil_data.csv")

X = df[["Nitrogen", "Phosphorus", "Potassium", "pH", "Moisture"]]
y = df["Soil_Quality"]

# Load model & encoder
with open("soil_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Train-test split (for metrics)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

class SoilInput(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    ph: float
    moisture: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: SoilInput):
    input_df = pd.DataFrame([[
        data.nitrogen,
        data.phosphorus,
        data.potassium,
        data.ph,
        data.moisture
    ]], columns=X.columns)

    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    label = le.inverse_transform([pred])[0]

    return {
        "soil_quality": label,
        "confidence": round(float(max(proba)), 2),
        "probabilities": {
            le.classes_[i]: round(float(p), 2)
            for i, p in enumerate(proba)
        }
    }

@app.get("/metrics")
def metrics():
    y_pred = model.predict(X_test)

    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 3),
        "precision": round(float(precision_score(y_test, y_pred, average="weighted")), 3),
        "recall": round(float(recall_score(y_test, y_pred, average="weighted")), 3),
        "f1_score": round(float(f1_score(y_test, y_pred, average="weighted")), 3),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "labels": list(le.classes_)
    }
