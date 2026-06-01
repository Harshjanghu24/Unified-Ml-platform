"""
Model Training Engine.
Trains appropriate models based on the detected problem type.
Includes hyperparameter tuning via GridSearchCV/RandomizedSearchCV
and K-Fold Cross Validation.

Supports XGBoost along with scikit-learn models.
"""

import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import warnings
warnings.filterwarnings("ignore")


# ── Hyperparameter grids ─────────────────────────────────────

PARAM_GRIDS = {
    "LogisticRegression": {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l2"],
        "solver": ["lbfgs"],
        "max_iter": [1000],
    },
    "RandomForestClassifier": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    },
    "DecisionTreeClassifier": {
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "criterion": ["gini", "entropy"],
    },
    "KNeighborsClassifier": {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"],
    },
    "XGBClassifier": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 6, 10],
        "learning_rate": [0.01, 0.1, 0.3],
    },
    "LinearRegression": {},  # No hyperparameters to tune
    "RandomForestRegressor": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    },
    "XGBRegressor": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 6, 10],
        "learning_rate": [0.01, 0.1, 0.3],
    },
}


def _evaluate_classifier(model, X_test, y_test, n_classes):
    """Compute classification metrics."""
    y_pred = model.predict(X_test)
    average = "binary" if n_classes == 2 else "weighted"

    # Dynamically determine pos_label for binary classification if labels are strings
    kwargs = {"average": average, "zero_division": 0}
    if average == "binary":
        unique_vals = list(set(y_test))
        if 1 in unique_vals:
            kwargs["pos_label"] = 1
        else:
            kwargs["pos_label"] = unique_vals[-1] if unique_vals else 1

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, **kwargs)), 4),
        "recall": round(float(recall_score(y_test, y_pred, **kwargs)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, **kwargs)), 4),
    }

    # ROC-AUC
    try:
        if n_classes == 2:
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
                metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_proba)), 4)
        else:
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)
                metrics["roc_auc"] = round(float(roc_auc_score(
                    y_test, y_proba, multi_class="ovr", average="weighted"
                )), 4)
    except Exception:
        metrics["roc_auc"] = None

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred).tolist()
    metrics["confusion_matrix"] = cm

    # Classification Report
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    # Convert numpy types to native Python
    clean_report = {}
    for k, v in report.items():
        if isinstance(v, dict):
            clean_report[k] = {kk: round(float(vv), 4) for kk, vv in v.items()}
        else:
            clean_report[k] = round(float(v), 4)
    metrics["classification_report"] = clean_report

    metrics["y_pred"] = y_pred.tolist()

    return metrics


def _evaluate_regressor(model, X_test, y_test):
    """Compute regression metrics."""
    y_pred = model.predict(X_test)

    metrics = {
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
        "mse": round(float(mean_squared_error(y_test, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        "r2_score": round(float(r2_score(y_test, y_pred)), 4),
        "y_pred": y_pred.tolist(),
        "y_test": y_test.tolist(),
    }

    return metrics


def _cross_validate(model, X_train, y_train, problem_type, cv=5):
    """Perform K-Fold Cross Validation."""
    if problem_type in ("binary", "multiclass"):
        scoring_metrics = {
            "accuracy": "accuracy",
            "f1": "f1_weighted",
        }
    else:
        scoring_metrics = {
            "r2": "r2",
            "neg_mae": "neg_mean_absolute_error",
        }

    cv_results = {}
    for name, scoring in scoring_metrics.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        cv_results[name] = {
            "scores": [round(float(s), 4) for s in scores],
            "mean": round(float(scores.mean()), 4),
            "std": round(float(scores.std()), 4),
        }

    return cv_results


def _tune_and_train(model_class, model_name, X_train, y_train, problem_type):
    """
    Perform hyperparameter tuning using GridSearchCV (small grids)
    or RandomizedSearchCV (large grids), then return the best model.
    """
    param_grid = PARAM_GRIDS.get(model_name, {})

    # XGBoost models need special kwargs
    extra_kwargs = {}
    if "XGB" in model_name:
        extra_kwargs = {
            "eval_metric": "logloss" if problem_type != "regression" else "rmse",
            "verbosity": 0, 
            "use_label_encoder": False
        }
        # Try to use GPU if available
        if HAS_TORCH:
            if torch.cuda.is_available():
                extra_kwargs["device"] = "cuda"
                extra_kwargs["tree_method"] = "hist"

    if not param_grid:
        # No tuning needed (e.g. LinearRegression)
        model = model_class(**extra_kwargs) if extra_kwargs else model_class()
        model.fit(X_train, y_train)
        return model, {}, None

    # Calculate grid size
    grid_size = 1
    for v in param_grid.values():
        grid_size *= len(v)

    scoring = "accuracy" if problem_type in ("binary", "multiclass") else "r2"

    base_model = model_class(**extra_kwargs) if extra_kwargs else model_class()

    if grid_size <= 50:
        search = GridSearchCV(
            base_model, param_grid, cv=3, scoring=scoring, n_jobs=-1, refit=True
        )
    else:
        search = RandomizedSearchCV(
            base_model, param_grid, n_iter=20, cv=3, scoring=scoring,
            n_jobs=-1, refit=True, random_state=42
        )

    search.fit(X_train, y_train)

    best_params = {k: (int(v) if isinstance(v, (np.integer,)) else
                       float(v) if isinstance(v, (np.floating,)) else v)
                   for k, v in search.best_params_.items()}

    return search.best_estimator_, best_params, round(float(search.best_score_), 4)


def train_models(X_train, X_test, y_train, y_test, problem_type):
    """
    Train all models for the given problem type with hyperparameter tuning and cross-validation.

    Returns:
        List of dicts, each containing:
          model_name, model_object, metrics, training_time, best_params, cv_scores, is_best
    """
    # Define models per problem type
    if problem_type == "binary":
        model_configs = [
            ("LogisticRegression", LogisticRegression),
            ("RandomForestClassifier", RandomForestClassifier),
        ]
        if HAS_XGBOOST:
            model_configs.append(("XGBClassifier", XGBClassifier))
    elif problem_type == "multiclass":
        model_configs = [
            ("DecisionTreeClassifier", DecisionTreeClassifier),
            ("RandomForestClassifier", RandomForestClassifier),
            ("KNeighborsClassifier", KNeighborsClassifier),
        ]
        if HAS_XGBOOST:
            model_configs.append(("XGBClassifier", XGBClassifier))
    else:  # regression
        model_configs = [
            ("LinearRegression", LinearRegression),
            ("RandomForestRegressor", RandomForestRegressor),
        ]
        if HAS_XGBOOST:
            model_configs.append(("XGBRegressor", XGBRegressor))

    results = []
    n_classes = int(y_train.nunique()) if problem_type != "regression" else 0

    for model_name, model_class in model_configs:
        start_time = time.time()

        # Hyperparameter tuning
        model, best_params, best_search_score = _tune_and_train(
            model_class, model_name, X_train, y_train, problem_type
        )

        training_time = round(time.time() - start_time, 3)

        # Evaluation
        if problem_type in ("binary", "multiclass"):
            metrics = _evaluate_classifier(model, X_test, y_test, n_classes)
        else:
            metrics = _evaluate_regressor(model, X_test, y_test)

        # Cross Validation
        cv_scores = _cross_validate(model, X_train, y_train, problem_type, cv=5)

        results.append({
            "model_name": model_name,
            "model_object": model,
            "metrics": metrics,
            "training_time": training_time,
            "best_params": best_params,
            "best_search_score": best_search_score,
            "cv_scores": cv_scores,
        })

    # Determine best model
    if problem_type in ("binary", "multiclass"):
        best_idx = max(range(len(results)), key=lambda i: results[i]["metrics"]["f1_score"])
    else:
        best_idx = max(range(len(results)), key=lambda i: results[i]["metrics"]["r2_score"])

    for i, r in enumerate(results):
        r["is_best"] = (i == best_idx)

    return results
