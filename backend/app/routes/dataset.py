"""
Dataset API Routes — Large Dataset Support.

Handles CSV upload (streaming to disk), dataset analysis, target
recommendations, and target selection.

Workflow:
  1. POST /upload-dataset     → Stream CSV to disk, load with tiered loader
  2. POST /analyze-columns    → Analyze all columns for target suitability
  3. GET  /target-recommendations → Get ranked target recommendations
  4. POST /select-target      → User selects a target column
"""

import os
import time

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..database import get_latest_dataset, save_dataset_record
from ..logger import setup_logger
from ..ml.analyzer import analyze_all_columns, get_target_recommendations
from ..ml.csv_loader import (
    TIER3_MAX_MB,
    CSVLoadResult,
    get_dataset_stats,
    load_csv,
)
from ..ml.detector import detect_problem_type, get_target_info

logger = setup_logger(__name__)

router = APIRouter(prefix="/api", tags=["Dataset"])

# In-memory store for the current dataset
_current_dataset = {}

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Streaming upload chunk size: 8 MB for efficient disk writes on large files
_UPLOAD_CHUNK_SIZE = 8 * 1024 * 1024


def _raise_for_csv_result(result: CSVLoadResult):
    """Convert a failed CSVLoadResult into an HTTPException with a user-friendly message."""
    if not result.success:
        raise HTTPException(status_code=400, detail=result.error or "Unknown CSV loading error.")


def _build_summary(df: pd.DataFrame, target_info: dict = None) -> dict:
    """Build a reusable dataset summary dict."""
    missing_values = {k: int(v) for k, v in df.isnull().sum().to_dict().items()}
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    summary = {
        "num_rows": len(df),
        "num_cols": len(df.columns),
        "columns": list(df.columns),
        "dtypes": dtypes,
        "missing_values": missing_values,
        "total_missing": int(df.isnull().sum().sum()),
        "numeric_columns": df.select_dtypes(
            include=["int64", "int32", "int16", "int8", "float64", "float32"]
        ).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=["object", "category"]).columns.tolist(),
    }
    if target_info:
        summary["target_info"] = target_info
    return summary


def get_current_dataset():
    """Access the current dataset state from other modules, auto-restoring from DB if needed."""
    if _current_dataset.get("df") is None:
        record = get_latest_dataset()
        if record:
            filename = record.get("filename")
            filepath = os.path.join(UPLOAD_DIR, filename)
            if os.path.exists(filepath):
                try:
                    result = load_csv(filepath)
                    if result.success:
                        _current_dataset["df"] = result.df
                        _current_dataset["filename"] = filename
                        _current_dataset["filepath"] = filepath
                        _current_dataset["target_column"] = record.get("target_column")
                        _current_dataset["problem_type"] = record.get("problem_type")
                        _current_dataset["dataset_id"] = record.get("id")
                        _current_dataset["dataset_stats"] = get_dataset_stats(result)
                    else:
                        logger.warning(f"Auto-restore failed: {result.error}")
                except Exception as e:
                    logger.warning(f"Failed to auto-restore dataset from disk: {e}")
    return _current_dataset


@router.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Step 1: Upload a CSV dataset WITHOUT specifying a target column.

    For ALL file sizes the upload is streamed to disk in chunks to
    prevent the server from buffering the entire request body in RAM.
    After saving, the tiered CSV loader reads the file with appropriate
    strategy based on file size.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    # ── Stream upload to disk ────────────────────────────────
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    upload_start = time.time()
    bytes_written = 0

    try:
        with open(filepath, "wb") as f:
            while chunk := await file.read(_UPLOAD_CHUNK_SIZE):
                f.write(chunk)
                bytes_written += len(chunk)
    except Exception as e:
        # Cleanup partial file on failure
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError:
                pass
        raise HTTPException(status_code=500, detail=f"Error saving uploaded file: {str(e)}") from e

    upload_duration = round(time.time() - upload_start, 2)
    file_size_mb = round(bytes_written / (1024 * 1024), 2)

    logger.info(
        f"Upload complete: {file.filename} — {file_size_mb:,.1f} MB "
        f"streamed to disk in {upload_duration}s"
    )

    # ── Pre-flight size check ────────────────────────────────
    if file_size_mb > TIER3_MAX_MB:
        # Clean up — file is too large
        os.remove(filepath)
        raise HTTPException(
            status_code=413,
            detail=(
                f"File is too large ({file_size_mb:,.1f} MB / {file_size_mb / 1024:.1f} GB). "
                f"Maximum allowed size is {TIER3_MAX_MB / 1024:.0f} GB."
            ),
        )

    # ── Load with tiered CSV loader ──────────────────────────
    # At upload time we don't know the target column or selected features yet,
    # so we load ALL columns.  For very large files (Tier 3), a re-load with
    # feature-aware usecols will happen at training time.
    result = load_csv(filepath)
    _raise_for_csv_result(result)
    df = result.df

    logger.info(
        f"Dataset loaded (Tier {result.tier}): {file.filename} — "
        f"{result.row_count:,} rows × {result.col_count} cols, "
        f"{result.memory_usage_mb} MB in-memory, "
        f"encoding={result.detected_encoding}, "
        f"delimiter={repr(result.detected_delimiter)}"
    )

    summary = _build_summary(df)

    # Preview (first 10 rows) - cast to object first to prevent pandas fillna type conflicts
    preview = df.head(10).astype(object).fillna("").to_dict(orient="records")

    # Store in memory (no target column yet)
    _current_dataset["df"] = df
    _current_dataset["filename"] = file.filename
    _current_dataset["filepath"] = filepath
    _current_dataset["target_column"] = None
    _current_dataset["problem_type"] = None
    _current_dataset["dataset_id"] = None
    _current_dataset["dataset_stats"] = get_dataset_stats(result)

    response = {
        "message": "Dataset uploaded successfully. Now analyze columns to find the best target.",
        "filename": file.filename,
        "summary": summary,
        "preview": preview,
        "dataset_stats": _current_dataset["dataset_stats"],
    }

    # Surface loader warnings to the frontend
    if result.warnings:
        response["warnings"] = result.warnings

    # Tier-specific messages
    if result.tier >= 2:
        response["message"] += (
            f" (Tier {result.tier}: memory-optimised loading applied, "
            f"{result.memory_savings_pct}% memory reduction)"
        )
    if result.sampling_applied:
        response["message"] += (
            f" Dataset sampled from {result.total_rows_in_file:,} to "
            f"{result.row_count:,} rows for training."
        )

    return response


@router.post("/analyze-columns")
async def analyze_columns():
    """
    Step 2: Analyze every column in the uploaded dataset.

    For each column computes:
      - Data Type, Unique Values, Missing %, Entropy, Variance, Cardinality
      - Target suitability scores for Binary, Multi-Class, and Regression
    """
    if _current_dataset.get("df") is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded. Upload a CSV first.")

    df = _current_dataset["df"]

    try:
        analysis = analyze_all_columns(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Column analysis failed: {str(e)}") from e

    _current_dataset["column_analysis"] = analysis

    return {
        "message": "Column analysis complete.",
        "total_columns": len(analysis),
        "analysis": analysis,
    }


@router.get("/target-recommendations")
async def target_recommendations():
    """
    Step 3: Get ranked target column recommendations.

    Returns columns grouped by suitability:
      - Best Candidates (Excellent/Good)
      - Good Candidates (Fair)
      - Poor Candidates (Poor/Unsuitable)
    """
    if _current_dataset.get("df") is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded.")

    df = _current_dataset["df"]

    try:
        recommendations = get_target_recommendations(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}") from e

    return recommendations


@router.post("/select-target")
async def select_target(target_column: str = Form(...)):
    """
    Step 4: User selects one of the recommended target columns.

    - Validates the column exists
    - Detects problem type
    - Saves dataset record to database
    """
    if _current_dataset.get("df") is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded.")

    df = _current_dataset["df"]

    if target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Column '{target_column}' not found. Available: {list(df.columns)}",
        )

    # Detect problem type
    problem_type = detect_problem_type(df, target_column)
    target_info = get_target_info(df, target_column)

    summary = _build_summary(df, target_info=target_info)

    # Save to database
    dataset_id = save_dataset_record(
        filename=_current_dataset["filename"],
        target_column=target_column,
        problem_type=problem_type,
        num_rows=len(df),
        num_cols=len(df.columns),
        summary=summary,
    )

    # Update in-memory state
    _current_dataset["target_column"] = target_column
    _current_dataset["problem_type"] = problem_type
    _current_dataset["dataset_id"] = dataset_id

    return {
        "message": (
            f"Target column '{target_column}' selected. Problem type: {problem_type.upper()}"
        ),
        "dataset_id": dataset_id,
        "target_column": target_column,
        "problem_type": problem_type,
        "target_info": target_info,
        "summary": summary,
    }


@router.get("/dataset-summary")
async def get_dataset_summary():
    """Return summary of the current (latest) dataset."""
    if _current_dataset.get("df") is None:
        # Try loading from database
        record = get_latest_dataset()
        if record:
            return {
                "summary": record.get("summary", {}),
                "problem_type": record.get("problem_type"),
                "has_target": record.get("target_column") is not None,
            }
        raise HTTPException(status_code=404, detail="No dataset uploaded yet.")

    df = _current_dataset["df"]
    target_column = _current_dataset.get("target_column")
    problem_type = _current_dataset.get("problem_type")

    target_info = None
    if target_column:
        target_info = get_target_info(df, target_column)
    summary = _build_summary(df, target_info=target_info)

    preview = df.head(10).astype(object).fillna("").to_dict(orient="records")

    return {
        "summary": summary,
        "problem_type": problem_type,
        "preview": preview,
        "dataset_id": _current_dataset.get("dataset_id"),
        "has_target": target_column is not None,
        "target_column": target_column,
        "dataset_stats": _current_dataset.get("dataset_stats"),
    }


@router.get("/dataset-columns")
async def get_columns():
    """Return column names for the uploaded dataset."""
    if _current_dataset.get("df") is None:
        raise HTTPException(status_code=404, detail="No dataset uploaded yet.")
    return {"columns": list(_current_dataset["df"].columns)}
