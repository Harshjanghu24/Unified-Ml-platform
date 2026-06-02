"""
AI-Powered Target Column Recommendation Engine.

Analyzes every column in a dataset and computes a suitability score
for each as a potential target variable. Recommends whether each column
is suitable for Binary Classification, Multi-Class Classification,
Regression, or is Unsuitable as a target.

Scoring Components:
  - Uniqueness Score
  - Distribution Score
  - Entropy Score
  - Missing Value Score
  - Cardinality Score
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy


def _compute_entropy(series: pd.Series) -> float:
    """Compute Shannon entropy for a series (higher = more uniform distribution)."""
    counts = series.value_counts(normalize=True)
    return float(scipy_entropy(counts, base=2))


def _compute_variance(series: pd.Series) -> float:
    """Compute variance for numeric series."""
    try:
        return float(series.dropna().var())
    except Exception:
        return 0.0


def _classify_column(col_stats: dict) -> dict:
    """
    Classify a column's suitability as a target for each problem type.

    Returns:
        dict with keys: binary_score, multiclass_score, regression_score,
                        suggested_type, confidence, recommendation
    """
    dtype = col_stats["dtype"]
    n_unique = col_stats["n_unique"]
    n_total = col_stats["n_total"]
    missing_pct = col_stats["missing_pct"]
    ent = col_stats["entropy"]
    is_numeric = col_stats["is_numeric"]
    cardinality_ratio = n_unique / max(n_total, 1)

    # ── Binary Classification Score ──
    binary_score = 0.0
    if n_unique == 2:
        binary_score = 80.0
        # Bonus for balanced distribution
        if ent > 0.8:
            binary_score += 15.0
        elif ent > 0.5:
            binary_score += 10.0
        else:
            binary_score += 5.0
        # Penalty for missing values
        binary_score -= missing_pct * 0.5
        binary_score = max(0.0, min(100.0, binary_score))

    # ── Multi-Class Classification Score ──
    multiclass_score = 0.0
    if 3 <= n_unique <= 20:
        multiclass_score = 60.0
        # Categorical columns are stronger candidates
        if not is_numeric:
            multiclass_score += 15.0
        elif pd.api.types.is_integer_dtype(dtype):
            multiclass_score += 10.0
        # Bonus for balanced entropy
        max_ent = np.log2(n_unique) if n_unique > 1 else 1
        ent_ratio = ent / max_ent if max_ent > 0 else 0
        if ent_ratio > 0.8:
            multiclass_score += 15.0
        elif ent_ratio > 0.5:
            multiclass_score += 10.0
        # Low cardinality ratio is good
        if cardinality_ratio < 0.05:
            multiclass_score += 5.0
        # Penalty for missing values
        multiclass_score -= missing_pct * 0.5
        multiclass_score = max(0.0, min(100.0, multiclass_score))

    # ── Regression Score ──
    regression_score = 0.0
    if is_numeric and n_unique > 20:
        regression_score = 60.0
        # More unique values → stronger regression candidate
        if n_unique > 100:
            regression_score += 15.0
        elif n_unique > 50:
            regression_score += 10.0
        else:
            regression_score += 5.0
        # Float types are better regression candidates
        if "float" in str(dtype):
            regression_score += 10.0
        # High variance is good
        if col_stats["variance"] > 0:
            regression_score += 5.0
        # Low missing values preferred
        regression_score -= missing_pct * 0.5
        # Very high cardinality (like IDs) is bad
        if cardinality_ratio > 0.95:
            regression_score -= 30.0
        regression_score = max(0.0, min(100.0, regression_score))

    # ── Unsuitable Detection ──
    is_unsuitable = False
    unsuitable_reason = ""

    # ID-like columns (all unique or nearly all unique)
    if cardinality_ratio > 0.9 and n_unique > 50:
        is_unsuitable = True
        unsuitable_reason = "Likely an ID/index column (too many unique values)"
        binary_score = 0
        multiclass_score = 0
        regression_score = max(0, regression_score - 40)

    # Too many missing values
    if missing_pct > 50:
        is_unsuitable = True
        unsuitable_reason = "Too many missing values (>50%)"
        binary_score *= 0.3
        multiclass_score *= 0.3
        regression_score *= 0.3

    # Only one unique value
    if n_unique <= 1:
        is_unsuitable = True
        unsuitable_reason = "Constant column (only 1 unique value)"
        binary_score = 0
        multiclass_score = 0
        regression_score = 0

    # Text/high-cardinality string columns
    if not is_numeric and n_unique > 20:
        is_unsuitable = True
        unsuitable_reason = "High-cardinality text column"
        binary_score = 0
        multiclass_score = 0
        regression_score = 0

    # Determine best type
    scores = {
        "Binary Classification": binary_score,
        "Multi-Class Classification": multiclass_score,
        "Regression": regression_score,
    }

    if is_unsuitable and max(scores.values()) < 20:
        suggested_type = "Unsuitable"
        confidence = 0.0
    else:
        suggested_type = max(scores, key=scores.get)
        confidence = max(scores.values())

    # Recommendation label
    if confidence >= 85:
        recommendation = "Excellent"
    elif confidence >= 70:
        recommendation = "Good"
    elif confidence >= 50:
        recommendation = "Fair"
    elif confidence > 20:
        recommendation = "Poor"
    else:
        recommendation = "Unsuitable"

    return {
        "binary_score": round(binary_score, 1),
        "multiclass_score": round(multiclass_score, 1),
        "regression_score": round(regression_score, 1),
        "suggested_type": suggested_type,
        "confidence": round(confidence, 1),
        "recommendation": recommendation,
        "is_unsuitable": is_unsuitable,
        "unsuitable_reason": unsuitable_reason,
    }


def analyze_all_columns(df: pd.DataFrame) -> list:
    """
    Analyze every column in the dataset for target suitability.

    For each column computes:
      - Data Type
      - Number of Unique Values
      - Missing Value Percentage
      - Class Distribution (top classes)
      - Entropy
      - Variance (numeric only)
      - Cardinality ratio
      - Target suitability scores

    Returns:
        Sorted list of column analysis dicts (best candidates first).
    """
    results = []

    for col in df.columns:
        series = df[col]
        n_total = len(series)
        n_unique = int(series.nunique())
        n_missing = int(series.isna().sum())
        missing_pct = round((n_missing / n_total) * 100, 2) if n_total > 0 else 0.0

        # Determine if numeric
        is_numeric = pd.api.types.is_numeric_dtype(series.dtype)

        # Entropy
        non_null = series.dropna()
        ent = _compute_entropy(non_null) if len(non_null) > 0 else 0.0

        # Variance
        var = _compute_variance(non_null) if is_numeric else 0.0

        # Class distribution (top 5)
        value_counts = non_null.value_counts()
        top_classes = {}
        for cls, count in value_counts.head(5).items():
            top_classes[str(cls)] = int(count)

        # Column stats
        col_stats = {
            "column_name": col,
            "dtype": str(series.dtype),
            "is_numeric": is_numeric,
            "n_unique": n_unique,
            "n_total": n_total,
            "n_missing": n_missing,
            "missing_pct": missing_pct,
            "entropy": round(ent, 4),
            "variance": round(var, 4) if is_numeric else None,
            "cardinality_ratio": round(n_unique / max(n_total, 1), 4),
            "top_classes": top_classes,
        }

        # Classify suitability
        classification = _classify_column(col_stats)
        col_stats.update(classification)

        results.append(col_stats)

    # Sort by confidence descending (best candidates first)
    results.sort(key=lambda x: x["confidence"], reverse=True)

    return results


def get_target_recommendations(df: pd.DataFrame) -> dict:
    """
    Generate a complete target recommendation report.

    Returns:
        dict with:
          - recommendations: sorted list of all column analyses
          - best_candidates: columns with 'Excellent' or 'Good' rating
          - good_candidates: columns with 'Fair' rating
          - poor_candidates: columns with 'Poor' or 'Unsuitable' rating
          - dataset_shape: (rows, cols)
    """
    all_analyses = analyze_all_columns(df)

    best = [a for a in all_analyses if a["recommendation"] in ("Excellent", "Good")]
    good = [a for a in all_analyses if a["recommendation"] == "Fair"]
    poor = [a for a in all_analyses if a["recommendation"] in ("Poor", "Unsuitable")]

    return {
        "recommendations": all_analyses,
        "best_candidates": best,
        "good_candidates": good,
        "poor_candidates": poor,
        "dataset_shape": {"rows": len(df), "cols": len(df.columns)},
        "total_columns": len(df.columns),
    }
