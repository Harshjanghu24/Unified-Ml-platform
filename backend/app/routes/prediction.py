"""
Prediction API Routes.
Handles single predictions (from form data) and batch predictions (from CSV upload).
"""

import os
import pandas as pd
import numpy as np
import joblib
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Dict, Any, Optional
from ..routes.dataset import get_current_dataset
from ..routes.training import get_training_state, MODELS_DIR
from ..ml.preprocessor import preprocess_for_prediction
from ..ml.explainability import explain_single_prediction

router = APIRouter(prefix="/api", tags=["Prediction"])


class PredictionInput(BaseModel):
    """Schema for single prediction input."""
    features: Dict[str, Any]
    explain: Optional[bool] = False


@router.post("/predict")
async def predict(input_data: PredictionInput):
    """
    Make a prediction using the best trained model.

    Accepts a dict of feature values, preprocesses them,
    and returns the prediction with optional SHAP explanation.
    """
    training_state = get_training_state()
    dataset = get_current_dataset()

    if not training_state.get("pipeline_path"):
        raise HTTPException(status_code=400, detail="No trained model available. Run training first.")

    # Load pipeline
    pipeline = joblib.load(training_state["pipeline_path"])
    feature_names = pipeline["feature_names"]
    scaler = pipeline["scaler"]
    label_encoder = pipeline["label_encoder"]
    problem_type = pipeline["problem_type"]

    # Find best model
    dataset_id = training_state["dataset_id"]
    best_model_name = training_state["best_model_name"]
    model_path = os.path.join(MODELS_DIR, f"{best_model_name}_{dataset_id}.joblib")

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Best model file not found.")

    model = joblib.load(model_path)

    # Preprocess input
    try:
        processed = preprocess_for_prediction(input_data.features, feature_names, scaler)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {str(e)}")

    # Predict
    prediction = model.predict(processed)
    result = {
        "model_used": best_model_name,
        "problem_type": problem_type,
    }

    if problem_type in ("binary", "multiclass"):
        predicted_class = int(prediction[0])
        if label_encoder:
            predicted_label = label_encoder.inverse_transform([predicted_class])[0]
        else:
            predicted_label = str(predicted_class)

        result["predicted_class"] = str(predicted_label)

        # Confidence score
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(processed)[0]
            result["confidence"] = round(float(max(proba)), 4)
            result["class_probabilities"] = {
                str(label_encoder.inverse_transform([i])[0] if label_encoder else str(i)):
                round(float(p), 4)
                for i, p in enumerate(proba)
            }
    else:
        result["predicted_value"] = round(float(prediction[0]), 4)

    # SHAP explanation
    if input_data.explain:
        try:
            X_train = dataset.get("df")
            if X_train is not None:
                from ..ml.preprocessor import build_preprocessing_pipeline
                prep = build_preprocessing_pipeline(X_train, pipeline["target_column"], problem_type)
                explanation = explain_single_prediction(
                    model, processed, prep["X_train"], feature_names, problem_type
                )
                result["explanation"] = explanation
        except Exception as e:
            result["explanation"] = {"error": str(e)}

    return result


@router.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    """
    Make batch predictions from an uploaded CSV file.
    Returns predictions for all rows.
    """
    training_state = get_training_state()

    if not training_state.get("pipeline_path"):
        raise HTTPException(status_code=400, detail="No trained model available.")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    # Read CSV
    content = await file.read()
    import io
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")

    # Load pipeline
    pipeline = joblib.load(training_state["pipeline_path"])
    feature_names = pipeline["feature_names"]
    scaler = pipeline["scaler"]
    label_encoder = pipeline["label_encoder"]
    problem_type = pipeline["problem_type"]

    # Load best model
    dataset_id = training_state["dataset_id"]
    best_model_name = training_state["best_model_name"]
    model_path = os.path.join(MODELS_DIR, f"{best_model_name}_{dataset_id}.joblib")
    model = joblib.load(model_path)

    # Preprocess each row
    results = []
    for idx, row in df.iterrows():
        try:
            processed = preprocess_for_prediction(row.to_dict(), feature_names, scaler)
            prediction = model.predict(processed)

            if problem_type in ("binary", "multiclass"):
                predicted_class = int(prediction[0])
                if label_encoder:
                    predicted_label = str(label_encoder.inverse_transform([predicted_class])[0])
                else:
                    predicted_label = str(predicted_class)

                row_result = {"row": idx, "predicted_class": predicted_label}
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(processed)[0]
                    row_result["confidence"] = round(float(max(proba)), 4)
            else:
                row_result = {"row": idx, "predicted_value": round(float(prediction[0]), 4)}

            results.append(row_result)
        except Exception as e:
            results.append({"row": idx, "error": str(e)})

    return {
        "model_used": best_model_name,
        "problem_type": problem_type,
        "total_rows": len(df),
        "predictions": results,
    }
