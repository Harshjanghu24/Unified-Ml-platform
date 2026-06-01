"""
Model Management API Routes.
Handles listing, saving, and deleting trained models.
Also provides the report generation endpoint.
"""

import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from ..database import get_all_models, delete_model_record, get_latest_dataset
from ..routes.training import get_training_state
from ..ml.report_generator import generate_report

router = APIRouter(prefix="/api", tags=["Models"])


@router.get("/models")
async def list_models():
    """Return all saved model records from the database."""
    models = get_all_models()
    return {"models": models}


@router.delete("/model/{model_id}")
async def delete_model(model_id: int):
    """Delete a model by ID (both database record and file)."""
    model_path = delete_model_record(model_id)
    if model_path is None:
        raise HTTPException(status_code=404, detail=f"Model with id {model_id} not found.")

    # Delete model file from disk
    if os.path.exists(model_path):
        os.remove(model_path)

    return {"message": f"Model {model_id} deleted successfully."}


@router.post("/generate-report")
async def generate_pdf_report():
    """
    Generate a downloadable PDF report containing:
      - Dataset summary
      - Preprocessing steps
      - Model comparison
      - Best model details
      - Feature selection results
      - Visualizations
    """
    training_state = get_training_state()
    if not training_state.get("models"):
        raise HTTPException(status_code=400, detail="No training results available.")

    dataset_record = get_latest_dataset()
    if not dataset_record:
        raise HTTPException(status_code=400, detail="No dataset found.")

    # Combine all plots
    all_plots = {}
    all_plots.update(training_state.get("eda_plots", {}))
    all_plots.update(training_state.get("eval_plots", {}))
    shap_results = training_state.get("shap_results", {})
    if shap_results.get("summary_plot"):
        all_plots["shap_summary"] = shap_results["summary_plot"]
    if shap_results.get("bar_plot"):
        all_plots["shap_feature_importance"] = shap_results["bar_plot"]

    try:
        filepath = generate_report(
            dataset_info=dataset_record,
            problem_type=training_state["problem_type"],
            preprocessing_steps=training_state.get("preprocessing_steps", []),
            model_results=training_state["models"],
            feature_selection=training_state.get("feature_selection"),
            plots=all_plots,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

    return FileResponse(
        filepath,
        media_type="application/pdf",
        filename=os.path.basename(filepath),
    )
