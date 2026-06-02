"""
Model Explainability Module using SHAP.
Generates SHAP summary plots and feature contribution explanations
to answer: "Why did the model predict this output?"
"""

import matplotlib
import numpy as np
import pandas as pd
import shap

matplotlib.use("Agg")
import base64
import warnings
from io import BytesIO

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def _fig_to_base64(fig):
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#0f172a")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


def generate_shap_explanations(
    model, X_train, X_test, feature_names, problem_type, max_samples=100
):
    """
    Generate SHAP explanations for the trained model.

    Returns:
        dict with keys:
          - summary_plot: base64 PNG of SHAP summary plot
          - feature_importance: list of {feature, mean_shap_value}
          - sample_explanations: SHAP values for first few test samples
    """
    results = {}

    # Subsample for performance
    if len(X_test) > max_samples:
        X_explain = (
            X_test.iloc[:max_samples] if isinstance(X_test, pd.DataFrame) else X_test[:max_samples]
        )
    else:
        X_explain = X_test

    if len(X_train) > 200:
        x_bg = X_train.iloc[:200] if isinstance(X_train, pd.DataFrame) else X_train[:200]
    else:
        x_bg = X_train

    try:
        # Choose appropriate explainer
        model_type = type(model).__name__

        if "Forest" in model_type or "Tree" in model_type:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_explain)
        elif "Linear" in model_type or "Logistic" in model_type:
            explainer = shap.LinearExplainer(model, x_bg)
            shap_values = explainer.shap_values(X_explain)
        else:
            # KNN and other models - use KernelExplainer with small background
            if len(x_bg) > 50:
                x_bg = x_bg.iloc[:50] if isinstance(x_bg, pd.DataFrame) else x_bg[:50]
            explainer = shap.KernelExplainer(model.predict, x_bg)
            shap_values = explainer.shap_values(X_explain, nsamples=50)

        # Handle multi-class SHAP values (list of arrays)
        if isinstance(shap_values, list):
            # For multi-class, average absolute SHAP values across classes
            shap_abs_mean = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            shap_for_plot = shap_values[1] if len(shap_values) == 2 else shap_values[0]
        else:
            shap_abs_mean = np.abs(shap_values).mean(axis=0)
            shap_for_plot = shap_values

        # ── SHAP Summary Plot ──
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_facecolor("#0f172a")
        fig.patch.set_facecolor("#0f172a")

        shap.summary_plot(
            shap_for_plot, X_explain, feature_names=feature_names, show=False, max_display=15
        )

        # Style the current figure
        current_fig = plt.gcf()
        current_fig.patch.set_facecolor("#0f172a")
        for ax in current_fig.get_axes():
            ax.set_facecolor("#0f172a")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            for spine in ax.spines.values():
                spine.set_color("#334155")

        results["summary_plot"] = _fig_to_base64(current_fig)

        # ── Feature Importance from SHAP ──
        importance_list = []
        for i, name in enumerate(feature_names):
            importance_list.append(
                {
                    "feature": name,
                    "mean_shap_value": round(float(shap_abs_mean[i]), 4),
                }
            )
        importance_list.sort(key=lambda x: x["mean_shap_value"], reverse=True)
        results["feature_importance"] = importance_list[:15]

        # ── SHAP Bar Plot ──
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        fig2.patch.set_facecolor("#0f172a")
        ax2.set_facecolor("#0f172a")

        top_features = importance_list[:15]
        names = [f["feature"] for f in top_features][::-1]
        values = [f["mean_shap_value"] for f in top_features][::-1]

        ax2.barh(names, values, color="#818cf8", edgecolor="#6366f1")
        ax2.set_xlabel("Mean |SHAP Value|", color="white", fontsize=12)
        ax2.set_title("SHAP Feature Importance", color="white", fontsize=14, fontweight="bold")
        ax2.tick_params(colors="white")
        for spine in ax2.spines.values():
            spine.set_color("#334155")

        results["bar_plot"] = _fig_to_base64(fig2)

    except Exception as e:
        results["error"] = str(e)
        results["summary_plot"] = None
        results["feature_importance"] = []
        results["bar_plot"] = None

    return results


def explain_single_prediction(model, instance, X_train, feature_names, problem_type):
    """
    Explain a single prediction using SHAP values.

    Returns:
        dict with feature contributions for the prediction.
    """
    try:
        if len(X_train) > 100:
            x_bg = X_train.iloc[:100] if isinstance(X_train, pd.DataFrame) else X_train[:100]
        else:
            x_bg = X_train

        model_type = type(model).__name__

        if "Forest" in model_type or "Tree" in model_type:
            explainer = shap.TreeExplainer(model)
        elif "Linear" in model_type or "Logistic" in model_type:
            explainer = shap.LinearExplainer(model, x_bg)
        else:
            if len(x_bg) > 50:
                x_bg = x_bg.iloc[:50] if isinstance(x_bg, pd.DataFrame) else x_bg[:50]
            explainer = shap.KernelExplainer(model.predict, x_bg)

        shap_values = explainer.shap_values(instance)

        if isinstance(shap_values, list):
            sv = shap_values[1] if len(shap_values) == 2 else shap_values[0]
        else:
            sv = shap_values

        if sv.ndim > 1:
            sv = sv[0]

        contributions = []
        for i, name in enumerate(feature_names):
            contributions.append(
                {
                    "feature": name,
                    "shap_value": round(float(sv[i]), 4),
                    "feature_value": round(
                        (
                            float(instance.iloc[0, i])
                            if isinstance(instance, pd.DataFrame)
                            else float(instance[0, i])
                        ),
                        4,
                    ),
                }
            )

        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return {"contributions": contributions[:10]}

    except Exception as e:
        return {"error": str(e), "contributions": []}
