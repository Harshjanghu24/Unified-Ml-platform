"""
Preprocessing API Routes.
Provides endpoints for automatic and manual dataset preprocessing,
returning detailed reports, statistics, and previews.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict

from ..routes.dataset import get_current_dataset
from ..ml.preprocessor import run_auto_preprocessing, run_manual_preprocessing, get_cardinality_recommendations
from ..logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter(prefix="/api/preprocess", tags=["Preprocessing"])


class PreprocessManualRequest(BaseModel):
    """Schema for manual preprocessing configuration."""

    missing_value_method: Optional[str] = "median"  # mean, median, mode, drop_rows, drop_cols
    duplicate_handling: Optional[str] = "remove"    # keep, remove
    encoding_method: Optional[str] = "onehot"       # fallback default
    column_encodings: Optional[Dict[str, str]] = None  # map of col -> "onehot" | "label" | "drop"
    scaling_method: Optional[str] = "standard"      # none, standard, minmax, robust
    outlier_method: Optional[str] = "none"          # none, iqr, zscore
    selected_features: Optional[List[str]] = None


@router.post("/auto")
async def preprocess_auto():
    """
    Apply automatic preprocessing on the current uploaded dataset.
    Uses the default preprocessing pipeline options.
    """
    dataset = get_current_dataset()
    if dataset.get("df") is None:
        raise HTTPException(
            status_code=400, detail="No dataset uploaded. Please upload a dataset first."
        )

    df = dataset["df"]
    target_column = dataset.get("target_column")
    problem_type = dataset.get("problem_type")

    if not target_column:
        raise HTTPException(
            status_code=400, detail="No target column selected. Please select a target first."
        )

    try:
        res = run_auto_preprocessing(df, target_column, problem_type)
        
        # Save preprocessed state
        dataset["processed_df"] = res["processed_df"]
        dataset["scaler"] = res["scaler"]
        dataset["categorical_encoders"] = res["categorical_encoders"]
        dataset["target_encoder"] = res["target_encoder"]
        dataset["feature_names"] = res["feature_names"]
        dataset["preprocessing_config"] = {
            "missing_value_method": "median",
            "duplicate_handling": "remove",
            "encoding_method": res["column_strategies"], # Dict of strategies
            "scaling_method": "standard",
            "outlier_method": "none",
            "selected_features": None
        }
        dataset["preprocessing_result"] = {
            "before_stats": res["before_stats"],
            "after_stats": res["after_stats"],
            "preprocessing_report": res["preprocessing_report"],
            "processing_summary": res["processing_summary"],
            "encoding_analysis": res["encoding_analysis"],
            "original_preview": res["original_preview"],
            "processed_preview": res["processed_preview"],
        }
        
        logger.info("Automatic preprocessing completed successfully.")
        return dataset["preprocessing_result"]
        
    except ValueError as e:
        logger.warning(f"OHE Dimensional Explosion aborted: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Automatic preprocessing failed")
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")


@router.post("/manual")
async def preprocess_manual(config: PreprocessManualRequest):
    """
    Apply user-configured manual preprocessing on the current uploaded dataset.
    """
    dataset = get_current_dataset()
    if dataset.get("df") is None:
        raise HTTPException(
            status_code=400, detail="No dataset uploaded. Please upload a dataset first."
        )

    df = dataset["df"]
    target_column = dataset.get("target_column")
    problem_type = dataset.get("problem_type")

    if not target_column:
        raise HTTPException(
            status_code=400, detail="No target column selected. Please select a target first."
        )

    try:
        config_dict = config.model_dump()
        res = run_manual_preprocessing(df, target_column, problem_type, config_dict)
        
        # Save preprocessed state
        dataset["processed_df"] = res["processed_df"]
        dataset["scaler"] = res["scaler"]
        dataset["categorical_encoders"] = res["categorical_encoders"]
        dataset["target_encoder"] = res["target_encoder"]
        dataset["feature_names"] = res["feature_names"]
        
        # Persistence structure
        dataset["preprocessing_config"] = config_dict
        dataset["preprocessing_config"]["encoding_method"] = res["column_strategies"] # Overwrite with the actual resolved dictionary mapping
        
        dataset["preprocessing_result"] = {
            "before_stats": res["before_stats"],
            "after_stats": res["after_stats"],
            "preprocessing_report": res["preprocessing_report"],
            "processing_summary": res["processing_summary"],
            "encoding_analysis": res["encoding_analysis"],
            "original_preview": res["original_preview"],
            "processed_preview": res["processed_preview"],
        }
        
        logger.info("Manual preprocessing completed successfully.")
        return dataset["preprocessing_result"]
        
    except ValueError as e:
        logger.warning(f"OHE Dimensional Explosion aborted: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Manual preprocessing failed")
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")


@router.get("/report")
async def get_preprocessing_report():
    """
    Retrieve the report and stats for the current preprocessed dataset.
    """
    dataset = get_current_dataset()
    if "preprocessing_result" not in dataset:
        raise HTTPException(
            status_code=404, detail="Dataset has not been preprocessed yet."
        )
    return dataset["preprocessing_result"]


@router.get("/recommendations")
async def get_encoding_recommendations():
    """
    Returns automated cardinality checks and encoding strategy recommendations for manual mode initialization.
    """
    dataset = get_current_dataset()
    if dataset.get("df") is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded.")
    df = dataset["df"]
    target_column = dataset.get("target_column")
    if not target_column:
        raise HTTPException(status_code=400, detail="No target column selected.")
        
    return get_cardinality_recommendations(df, target_column)
