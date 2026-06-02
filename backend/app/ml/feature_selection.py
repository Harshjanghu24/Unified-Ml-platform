"""
Feature Selection Module.
Implements Chi-Square Test, Mutual Information, and Random Forest Feature Importance
for automatic feature ranking and selection.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    chi2,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.preprocessing import MinMaxScaler


def chi_square_selection(X_train, y_train, feature_names, top_n=10):
    """
    Perform Chi-Square test for feature selection (classification only).
    Requires non-negative features, so applies MinMaxScaler first.

    Returns:
        List of dicts with 'feature', 'chi2_score', 'p_value'.
    """
    scaler = MinMaxScaler()
    X_positive = scaler.fit_transform(X_train)

    chi2_scores, p_values = chi2(X_positive, y_train)

    results = []
    for i, name in enumerate(feature_names):
        c_score = float(chi2_scores[i])
        p_val = float(p_values[i])
        results.append({
            "feature": name,
            "chi2_score": round(c_score, 4) if not np.isnan(c_score) else 0.0,
            "p_value": round(p_val, 6) if not np.isnan(p_val) else 1.0,
        })

    results.sort(key=lambda x: x["chi2_score"], reverse=True)
    return results[:top_n]


def mutual_information_selection(X_train, y_train, feature_names, problem_type, top_n=10):
    """
    Compute Mutual Information scores for feature ranking.
    Uses mutual_info_classif for classification, mutual_info_regression for regression.

    Returns:
        List of dicts with 'feature', 'mi_score'.
    """
    if problem_type in ("binary", "multiclass"):
        mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
    else:
        mi_scores = mutual_info_regression(X_train, y_train, random_state=42)

    results = []
    for i, name in enumerate(feature_names):
        mi_score = float(mi_scores[i])
        results.append({
            "feature": name,
            "mi_score": round(mi_score, 4) if not np.isnan(mi_score) else 0.0,
        })

    results.sort(key=lambda x: x["mi_score"], reverse=True)
    return results[:top_n]


def random_forest_importance(X_train, y_train, feature_names, problem_type, top_n=10):
    """
    Use Random Forest to determine feature importances.

    Returns:
        List of dicts with 'feature', 'importance'.
    """
    if problem_type in ("binary", "multiclass"):
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    model.fit(X_train, y_train)
    importances = model.feature_importances_

    results = []
    for i, name in enumerate(feature_names):
        importance = float(importances[i])
        results.append({
            "feature": name,
            "importance": round(importance, 4) if not np.isnan(importance) else 0.0,
        })

    results.sort(key=lambda x: x["importance"], reverse=True)
    return results[:top_n]


def run_feature_selection(X_train, y_train, feature_names, problem_type, top_n=10):
    """
    Run all applicable feature selection methods and return combined results.

    Returns:
        dict with keys: 'chi_square', 'mutual_information', 'random_forest_importance'
    """
    results = {}

    # Chi-Square (classification only)
    if problem_type in ("binary", "multiclass"):
        try:
            results["chi_square"] = chi_square_selection(X_train, y_train, feature_names, top_n)
        except Exception as e:
            results["chi_square"] = {"error": str(e)}

    # Mutual Information (works for all problem types)
    try:
        results["mutual_information"] = mutual_information_selection(
            X_train, y_train, feature_names, problem_type, top_n
        )
    except Exception as e:
        results["mutual_information"] = {"error": str(e)}

    # Random Forest Feature Importance
    try:
        results["random_forest_importance"] = random_forest_importance(
            X_train, y_train, feature_names, problem_type, top_n
        )
    except Exception as e:
        results["random_forest_importance"] = {"error": str(e)}

    return results
