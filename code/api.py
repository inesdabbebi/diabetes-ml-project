"""
============================================================
  api.py  —  FastAPI REST API for Diabetes Risk Prediction
============================================================
Run:
    uvicorn code.api:app --reload
Docs:
    http://localhost:8000/docs
"""

import io
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# ── App ──────────────────────────────────────────────────
app = FastAPI(
    title       = "GlycoScan — Diabetes Risk Prediction API",
    description = "Predict diabetes risk using Logistic Regression, KNN, and Random Forest.",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Models (once at startup) ────────────────────────
MODELS_DIR = "models"

try:
    scaler         = joblib.load(f"{MODELS_DIR}/scaler.pkl")
    label_encoders = joblib.load(f"{MODELS_DIR}/label_encoders.pkl")
    models = {
        "Logistic Regression": joblib.load(f"{MODELS_DIR}/Logistic_Regression.pkl"),
        "KNN"                : joblib.load(f"{MODELS_DIR}/KNN.pkl"),
        "Random Forest"      : joblib.load(f"{MODELS_DIR}/Random_Forest.pkl"),
    }
    EXPECTED_COLS = [
        "Gender", "Age", "Physical Activity", "Smoking Status",
        "Alcohol Intake", "Glucose", "Blood Pressure", "Skin Thickness",
        "Insulin", "BMI", "Cholesterol", "Diabetes Pedigree Function",
        "Family History", "Hypertension"
    ]
    models_loaded = True
    print("All models loaded successfully.")
except Exception as e:
    models_loaded = False
    EXPECTED_COLS = []
    print(f"Error loading models: {e}")


# ════════════════════════════════════════════════════════
#  INPUT SCHEMA
# ════════════════════════════════════════════════════════
class PatientData(BaseModel):
    Gender                    : str
    Age                       : int
    Physical_Activity         : str
    Smoking_Status            : str
    Alcohol_Intake            : Optional[str] = None
    Glucose                   : float
    Blood_Pressure            : float
    Skin_Thickness            : float
    Insulin                   : float
    BMI                       : float
    Cholesterol               : float
    Diabetes_Pedigree_Function: float
    Family_History            : str
    Hypertension              : str

    class Config:
        json_schema_extra = {
            "example": {
                "Gender"                    : "Male",
                "Age"                       : 45,
                "Physical_Activity"         : "Moderate",
                "Smoking_Status"            : "Never",
                "Alcohol_Intake"            : None,
                "Glucose"                   : 120.0,
                "Blood_Pressure"            : 70.0,
                "Skin_Thickness"            : 20.0,
                "Insulin"                   : 80.0,
                "BMI"                       : 28.5,
                "Cholesterol"               : 190.0,
                "Diabetes_Pedigree_Function": 0.5,
                "Family_History"            : "No",
                "Hypertension"              : "No",
            }
        }


# ════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════
def encode_value(le, value):
    """Encode a single value using a LabelEncoder, handling NaN safely."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        for i, c in enumerate(le.classes_):
            try:
                if pd.isna(c):
                    return i
            except Exception:
                pass
        return 0
    try:
        return int(le.transform([value])[0])
    except ValueError:
        valid = [c for c in le.classes_ if not (isinstance(c, float) and np.isnan(c))]
        raise HTTPException(
            status_code=422,
            detail=f"Invalid value '{value}'. Valid options: {valid}"
        )


def preprocess(data: dict) -> np.ndarray:
    """Convert API input dict to scaled numpy array matching training format."""
    row = {
        "Gender"                    : data.get("Gender"),
        "Age"                       : data.get("Age"),
        "Physical Activity"         : data.get("Physical_Activity"),
        "Smoking Status"            : data.get("Smoking_Status"),
        "Alcohol Intake"            : data.get("Alcohol_Intake"),
        "Glucose"                   : data.get("Glucose"),
        "Blood Pressure"            : data.get("Blood_Pressure"),
        "Skin Thickness"            : data.get("Skin_Thickness"),
        "Insulin"                   : data.get("Insulin"),
        "BMI"                       : data.get("BMI"),
        "Cholesterol"               : data.get("Cholesterol"),
        "Diabetes Pedigree Function": data.get("Diabetes_Pedigree_Function"),
        "Family History"            : data.get("Family_History"),
        "Hypertension"              : data.get("Hypertension"),
    }

    df = pd.DataFrame([row])

    cat_cols = [
        "Gender", "Physical Activity", "Smoking Status",
        "Alcohol Intake", "Family History", "Hypertension"
    ]

    for col in cat_cols:
        if col in label_encoders:
            df[col] = encode_value(label_encoders[col], df[col].values[0])

    df = df[EXPECTED_COLS].astype(float)
    return scaler.transform(df)


# ════════════════════════════════════════════════════════
#  ENDPOINTS
# ════════════════════════════════════════════════════════

@app.get("/", tags=["System"])
def root():
    return {
        "message": "GlycoScan API is running.",
        "docs"   : "http://localhost:8000/docs",
        "health" : "http://localhost:8000/health",
    }


@app.get("/health", tags=["System"])
def health():
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded.")
    return {
        "status"          : "ok",
        "models"          : list(models.keys()),
        "expected_columns": EXPECTED_COLS,
        "version"         : "1.0.0",
    }


@app.post("/predict", tags=["Prediction"])
def predict(patient: PatientData):
    """Predict diabetes risk for a single patient from all 3 models."""
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded.")

    X = preprocess(patient.dict())

    results = {}
    for name, model in models.items():
        pred  = int(model.predict(X)[0])
        proba = float(model.predict_proba(X)[0][1])
        results[name] = {
            "prediction"  : pred,
            "label"       : "Diabetic" if pred == 1 else "Non-diabetic",
            "probability" : round(proba, 4),
            "risk_percent": round(proba * 100, 1),
        }

    avg_proba  = round(sum(r["probability"] for r in results.values()) / len(results), 4)
    final_pred = 1 if avg_proba >= 0.5 else 0

    return {
        "patient_data": patient.dict(),
        "models"      : results,
        "ensemble"    : {
            "prediction"  : final_pred,
            "label"       : "Diabetic" if final_pred == 1 else "Non-diabetic",
            "probability" : avg_proba,
            "risk_percent": round(avg_proba * 100, 1),
        },
    }


@app.post("/predict_batch", tags=["Prediction"])
async def predict_batch(file: UploadFile = File(...)):
    """Predict diabetes risk for multiple patients from a CSV file."""
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded.")
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a .csv")

    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    cat_cols = [
        "Gender", "Physical Activity", "Smoking Status",
        "Alcohol Intake", "Family History", "Hypertension"
    ]

    for col in cat_cols:
        if col in df.columns and col in label_encoders:
            le = label_encoders[col]
            df[col] = df[col].apply(lambda x: encode_value(le, x))

    X      = scaler.transform(df[EXPECTED_COLS].astype(float))
    model  = models["Random Forest"]
    preds  = model.predict(X).tolist()
    probas = model.predict_proba(X)[:, 1].tolist()

    results = [
        {
            "row"         : i + 1,
            "prediction"  : p,
            "label"       : "Diabetic" if p == 1 else "Non-diabetic",
            "probability" : round(pr, 4),
            "risk_percent": round(pr * 100, 1),
        }
        for i, (p, pr) in enumerate(zip(preds, probas))
    ]

    return {
        "total_patients"    : len(results),
        "diabetic_count"    : sum(1 for r in results if r["prediction"] == 1),
        "non_diabetic_count": sum(1 for r in results if r["prediction"] == 0),
        "predictions"       : results,
    }