"""
Automatic Problem Type Detector.
Analyzes the target column of a dataset to determine whether
the problem is Binary Classification, Multi-Class Classification, or Regression.
"""

import pandas as pd
import numpy as np


def detect_problem_type(df: pd.DataFrame, target_column: str) -> str:
    """
    Detect the supervised learning problem type from the target column.

    Rules:
      - If target is non-numeric (object/category) with exactly 2 unique → 'binary'
      - If target is non-numeric with > 2 unique → 'multiclass'
      - If target is numeric with ≤ 2 unique → 'binary'
      - If target is numeric with 3–20 unique and integer-like → 'multiclass'
      - If target is numeric with > 20 unique or float → 'regression'

    Returns:
        One of: 'binary', 'multiclass', 'regression'
    """
    target = df[target_column].dropna()
    n_unique = target.nunique()
    dtype = target.dtype

    # Categorical / object columns
    if dtype == "object" or pd.api.types.is_categorical_dtype(dtype):
        if n_unique == 2:
            return "binary"
        return "multiclass"

    # Numeric columns
    if n_unique <= 2:
        return "binary"

    # Integer-like with small cardinality → classification
    if n_unique <= 20 and pd.api.types.is_integer_dtype(dtype):
        return "multiclass"

    # Check if float values are actually class labels (e.g. 0.0, 1.0, 2.0)
    if n_unique <= 20:
        if np.all(target == target.astype(int)):
            return "multiclass"

    return "regression"


def get_target_info(df: pd.DataFrame, target_column: str) -> dict:
    """
    Return detailed information about the target column.
    """
    target = df[target_column].dropna()
    problem_type = detect_problem_type(df, target_column)

    info = {
        "column_name": target_column,
        "problem_type": problem_type,
        "dtype": str(target.dtype),
        "n_unique": int(target.nunique()),
        "missing_count": int(df[target_column].isna().sum()),
        "total_count": len(df),
    }

    if problem_type in ("binary", "multiclass"):
        value_counts = target.value_counts().to_dict()
        # Convert numpy types to native Python types for JSON serialization
        info["class_distribution"] = {str(k): int(v) for k, v in value_counts.items()}
        info["classes"] = [str(c) for c in sorted(target.unique())]
    else:
        info["min"] = float(target.min())
        info["max"] = float(target.max())
        info["mean"] = float(target.mean())
        info["median"] = float(target.median())
        info["std"] = float(target.std())

    return info
