# """
# Dataset API Routes.
# Handles CSV upload, dataset analysis, target recommendations, and target selection.

# Workflow:
#   1. POST /upload-dataset     → Upload CSV (no target column needed)
#   2. POST /analyze-columns    → Analyze all columns for target suitability
#   3. GET  /target-recommendations → Get ranked target recommendations
#   4. POST /select-target      → User selects a target column
# """

import os
import pandas as pd
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from ..database import save_dataset_record, get_latest_dataset
from ..ml.detector import detect_problem_type, get_target_info
from ..ml.analyzer import analyze_all_columns, get_target_recommendations

router = APIRouter(prefix="/api", tags=["Dataset"])

# In-memory store for the current dataset
_current_dataset = {}

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def get_current_dataset():
    """Access the current dataset state from other modules, auto-restoring from DB if needed."""
    if _current_dataset.get("df") is None:
        record = get_latest_dataset()
        if record:
            filename = record.get("filename")
            filepath = os.path.join(UPLOAD_DIR, filename)
            if os.path.exists(filepath):
                try:
                    df = None
                    for encoding in ["utf-8", "latin1", "cp1252", "utf-8-sig"]:
                        try:
                            df = pd.read_csv(filepath, encoding=encoding, low_memory=False)
                            break
                        except Exception:
                            continue
                    if df is not None:
                        _current_dataset["df"] = df
                        _current_dataset["filename"] = filename
                        _current_dataset["filepath"] = filepath
                        _current_dataset["target_column"] = record.get("target_column")
                        _current_dataset["problem_type"] = record.get("problem_type")
                        _current_dataset["dataset_id"] = record.get("id")
                except Exception as e:
                    print(f"[WARNING] Failed to auto-restore dataset from disk: {e}")
    return _current_dataset


@router.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Step 1: Upload a CSV dataset WITHOUT specifying a target column.

    - Saves the file to disk streaming in chunks to prevent high memory usage
    - Reads it with Pandas trying multiple common encodings
    - Computes basic summary statistics
    - Stores in memory for subsequent analysis
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    # Save file in chunks to prevent high memory usage
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(filepath, "wb") as f:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                f.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving uploaded file: {str(e)}") from e

    # Read dataset with fallback encodings to support non-UTF-8 datasets (e.g. latin1, cp1252)
    df = None
    encodings_to_try = ["utf-8", "latin1", "cp1252", "utf-8-sig"]
    last_error = None
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(filepath, encoding=encoding, low_memory=False)
            break
        except UnicodeDecodeError as e:
            last_error = e
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}") from e

    if df is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unicode decode error. Failed to read CSV with common encodings. Detail: {str(last_error)}"
        )

    if df.empty:
        raise HTTPException(status_code=400, detail="The uploaded CSV file is empty.")

    # Dataset summary
    missing_values = {k: int(v) for k, v in df.isnull().sum().to_dict().items()}
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    summary = {
        "num_rows": len(df),
        "num_cols": len(df.columns),
        "columns": list(df.columns),
        "dtypes": dtypes,
        "missing_values": missing_values,
        "total_missing": int(df.isnull().sum().sum()),
        "numeric_columns": df.select_dtypes(include=["int64", "float64"]).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=["object", "category"]).columns.tolist(),
    }

    # Preview (first 10 rows) - cast to object first to prevent pandas fillna type conflicts
    preview = df.head(10).astype(object).fillna("").to_dict(orient="records")

    # Store in memory (no target column yet)
    _current_dataset["df"] = df
    _current_dataset["filename"] = file.filename
    _current_dataset["filepath"] = filepath
    _current_dataset["target_column"] = None
    _current_dataset["problem_type"] = None
    _current_dataset["dataset_id"] = None

    return {
        "message": "Dataset uploaded successfully. Now analyze columns to find the best target.",
        "filename": file.filename,
        "summary": summary,
        "preview": preview,
    }


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
            detail=f"Column '{target_column}' not found. Available: {list(df.columns)}"
        )

    # Detect problem type
    problem_type = detect_problem_type(df, target_column)
    target_info = get_target_info(df, target_column)

    # Dataset summary
    missing_values = {k: int(v) for k, v in df.isnull().sum().to_dict().items()}
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    summary = {
        "num_rows": len(df),
        "num_cols": len(df.columns),
        "columns": list(df.columns),
        "dtypes": dtypes,
        "missing_values": missing_values,
        "total_missing": int(df.isnull().sum().sum()),
        "target_info": target_info,
        "numeric_columns": df.select_dtypes(include=["int64", "float64"]).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=["object", "category"]).columns.tolist(),
    }

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
        "message": f"Target column '{target_column}' selected. Problem type: {problem_type.upper()}",
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

    missing_values = {k: int(v) for k, v in df.isnull().sum().to_dict().items()}
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    summary = {
        "num_rows": len(df),
        "num_cols": len(df.columns),
        "columns": list(df.columns),
        "dtypes": dtypes,
        "missing_values": missing_values,
        "total_missing": int(df.isnull().sum().sum()),
        "numeric_columns": df.select_dtypes(include=["int64", "float64"]).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=["object", "category"]).columns.tolist(),
    }

    if target_column:
        target_info = get_target_info(df, target_column)
        summary["target_info"] = target_info

    preview = df.head(10).fillna("").to_dict(orient="records")

    return {
        "summary": summary,
        "problem_type": problem_type,
        "preview": preview,
        "dataset_id": _current_dataset.get("dataset_id"),
        "has_target": target_column is not None,
        "target_column": target_column,
    }


@router.get("/dataset-columns")
async def get_columns():
    """Return column names for the uploaded dataset."""
    if _current_dataset.get("df") is None:
        raise HTTPException(status_code=404, detail="No dataset uploaded yet.")
    return {"columns": list(_current_dataset["df"].columns)}
