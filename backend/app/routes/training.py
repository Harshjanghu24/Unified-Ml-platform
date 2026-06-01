"""
Training API Routes.
Handles model training, preprocessing, feature selection,
hyperparameter tuning, and cross-validation orchestration.
"""

import os
import joblib
from fastapi import APIRouter, HTTPException
from ..routes.dataset import get_current_dataset
from ..ml.preprocessor import build_preprocessing_pipeline
from ..ml.trainer import train_models
from ..ml.feature_selection import run_feature_selection
from ..ml.explainability import generate_shap_explanations
from ..ml.eda import (
    generate_confusion_matrix_plot, generate_roc_curve,
    generate_actual_vs_predicted, generate_residual_plot,
    generate_all_eda
)
from ..database import save_model_record

router = APIRouter(prefix="/api", tags=["Training"])

# In-memory store for training results
_training_state = {}


def get_training_state():
    """Access training state from other modules."""
    return _training_state


MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


@router.post("/train")
async def train():
    """
    Execute the full ML pipeline:
      1. Preprocess data
      2. Feature selection
      3. Train models with hyperparameter tuning
      4. Cross-validation
      5. SHAP explainability
      6. Generate evaluation plots
      7. Save results to database
    """
    dataset = get_current_dataset()
    if dataset.get("df") is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded. Please upload a dataset first.")

    df = dataset["df"]
    target_column = dataset["target_column"]
    problem_type = dataset["problem_type"]
    dataset_id = dataset["dataset_id"]

    # ── Step 1: Preprocessing ──
    try:
        prep_result = build_preprocessing_pipeline(df, target_column)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")

    X_train = prep_result["X_train"]
    X_test = prep_result["X_test"]
    y_train = prep_result["y_train"]
    y_test = prep_result["y_test"]
    feature_names = prep_result["feature_names"]
    scaler = prep_result["scaler"]
    label_encoder = prep_result["label_encoder"]
    steps_log = prep_result["steps_log"]

    # ── Step 2: Feature Selection ──
    try:
        feature_selection = run_feature_selection(
            X_train, y_train, feature_names, problem_type, top_n=15
        )
    except Exception as e:
        feature_selection = {"error": str(e)}

    # ── Step 3: Train Models ──
    try:
        model_results = train_models(X_train, X_test, y_train, y_test, problem_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")

    # ── Step 4: Generate evaluation plots ──
    eval_plots = {}
    best_model_result = next((r for r in model_results if r["is_best"]), model_results[0])
    best_model = best_model_result["model_object"]

    if problem_type in ("binary", "multiclass"):
        # Confusion matrix for best model
        cm = best_model_result["metrics"].get("confusion_matrix")
        if cm:
            classes = None
            if label_encoder:
                classes = [str(c) for c in label_encoder.classes_]
            eval_plots["confusion_matrix"] = generate_confusion_matrix_plot(cm, classes)

        # ROC Curve
        n_classes = int(y_test.nunique())
        roc = generate_roc_curve(best_model, X_test, y_test, n_classes)
        if roc:
            eval_plots["roc_curve"] = roc
    else:
        # Actual vs Predicted
        y_pred = best_model_result["metrics"].get("y_pred", [])
        y_actual = best_model_result["metrics"].get("y_test", y_test.tolist())
        if y_pred:
            eval_plots["actual_vs_predicted"] = generate_actual_vs_predicted(y_actual, y_pred)
            eval_plots["residual_plot"] = generate_residual_plot(y_actual, y_pred)

    # ── Step 5: SHAP Explainability ──
    try:
        shap_results = generate_shap_explanations(
            best_model, X_train, X_test, feature_names, problem_type
        )
    except Exception as e:
        shap_results = {"error": str(e)}

    # ── Step 6: EDA ──
    try:
        eda_plots = generate_all_eda(df, target_column, problem_type)
    except Exception as e:
        eda_plots = {"error": str(e)}

    # ── Step 7: Save models to disk and database ──
    saved_models = []
    for result in model_results:
        model_filename = f"{result['model_name']}_{dataset_id}.joblib"
        model_path = os.path.join(MODELS_DIR, model_filename)
        joblib.dump(result["model_object"], model_path)

        # Clean metrics for storage (remove large arrays)
        clean_metrics = {k: v for k, v in result["metrics"].items()
                        if k not in ("y_pred", "y_test")}

        model_id = save_model_record(
            dataset_id=dataset_id,
            model_name=result["model_name"],
            problem_type=problem_type,
            metrics=clean_metrics,
            training_time=result["training_time"],
            model_path=model_path,
            is_best=result["is_best"],
            best_params=result.get("best_params"),
            cv_scores=result.get("cv_scores"),
        )

        saved_models.append({
            "model_id": model_id,
            "model_name": result["model_name"],
            "is_best": result["is_best"],
            "metrics": clean_metrics,
            "training_time": result["training_time"],
            "best_params": result.get("best_params", {}),
            "best_search_score": result.get("best_search_score"),
            "cv_scores": result.get("cv_scores", {}),
        })

    # Save pipeline artifacts for prediction
    pipeline_path = os.path.join(MODELS_DIR, f"pipeline_{dataset_id}.joblib")
    joblib.dump({
        "scaler": scaler,
        "label_encoder": label_encoder,
        "feature_names": feature_names,
        "problem_type": problem_type,
        "target_column": target_column,
    }, pipeline_path)

    # Store everything in memory
    _training_state["models"] = saved_models
    _training_state["preprocessing_steps"] = steps_log
    _training_state["feature_selection"] = feature_selection
    _training_state["eval_plots"] = eval_plots
    _training_state["eda_plots"] = eda_plots
    _training_state["shap_results"] = shap_results
    _training_state["problem_type"] = problem_type
    _training_state["dataset_id"] = dataset_id
    _training_state["best_model_name"] = best_model_result["model_name"]
    _training_state["pipeline_path"] = pipeline_path

    return {
        "message": "Training completed successfully",
        "problem_type": problem_type,
        "models": saved_models,
        "preprocessing_steps": steps_log,
        "feature_selection": feature_selection,
        "best_model": best_model_result["model_name"],
    }


@router.get("/metrics")
async def get_metrics():
    """Return all training metrics, plots, and SHAP results."""
    if not _training_state.get("models"):
        raise HTTPException(status_code=404, detail="No models trained yet. Run /api/train first.")

    return {
        "models": _training_state["models"],
        "preprocessing_steps": _training_state.get("preprocessing_steps", []),
        "feature_selection": _training_state.get("feature_selection", {}),
        "eval_plots": _training_state.get("eval_plots", {}),
        "shap_results": {
            k: v for k, v in _training_state.get("shap_results", {}).items()
            if k != "error" or v is not None
        },
        "problem_type": _training_state.get("problem_type"),
        "best_model": _training_state.get("best_model_name"),
    }


@router.get("/eda")
async def get_eda():
    """Return EDA plots."""
    if not _training_state.get("eda_plots"):
        # If training hasn't been run yet but dataset exists, generate EDA
        dataset = get_current_dataset()
        if dataset.get("df") is not None:
            try:
                eda_plots = generate_all_eda(
                    dataset["df"], dataset["target_column"], dataset["problem_type"]
                )
                return {"plots": eda_plots}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"EDA generation failed: {str(e)}")
        raise HTTPException(status_code=404, detail="No dataset or training data available.")

    return {"plots": _training_state["eda_plots"]}
