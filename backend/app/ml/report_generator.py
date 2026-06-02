"""
Auto Report Generation Module.
Generates a downloadable PDF report containing:
  - Dataset summary
  - Problem type
  - Preprocessing steps
  - Model comparison table
  - Best model details with metrics
  - Feature importance
  - Key visualizations
"""

import base64
import os
import tempfile
from datetime import datetime

from fpdf import FPDF


class MLReportPDF(FPDF):
    """Custom PDF class with branded header/footer."""

    def header(self):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(79, 70, 229)  # Indigo
        self.cell(
            0, 10, "Unified Supervised Learning Platform", align="C", new_x="LMARGIN", new_y="NEXT"
        )
        self.set_draw_color(79, 70, 229)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(
            0,
            10,
            (
                f"Page {self.page_no()}/{{nb}} | "
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            ),
            align="C",
        )

    def section_title(self, title):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(30, 41, 59)
        self.ln(5)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(79, 70, 229)
        self.line(10, self.get_y(), 80, self.get_y())
        self.ln(3)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(51, 65, 85)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def key_value(self, key, value):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(30, 41, 59)
        self.cell(60, 7, f"{key}:", new_x="RIGHT")
        self.set_font("Helvetica", "", 10)
        self.set_text_color(71, 85, 105)
        self.cell(0, 7, str(value), new_x="LMARGIN", new_y="NEXT")


def generate_report(
    dataset_info: dict,
    problem_type: str,
    preprocessing_steps: list,
    model_results: list,
    feature_selection: dict = None,
    plots: dict = None,
    output_dir: str = None,
) -> str:
    """
    Generate a comprehensive PDF report.

    Args:
        dataset_info: dict with filename, num_rows, num_cols, target_column, etc.
        problem_type: 'binary', 'multiclass', or 'regression'
        preprocessing_steps: list of step description strings
        model_results: list of model result dicts from trainer
        feature_selection: dict of feature selection results
        plots: dict of base64-encoded plot images
        output_dir: directory to save the PDF

    Returns:
        Absolute path to the generated PDF file.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "reports")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ml_report_{timestamp}.pdf"
    filepath = os.path.join(output_dir, filename)

    pdf = MLReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ── Title ──
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 15, "Machine Learning Analysis Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    problem_labels = {
        "binary": "Binary Classification",
        "multiclass": "Multi-Class Classification",
        "regression": "Regression",
    }

    # ── 1. Dataset Summary ──
    pdf.section_title("1. Dataset Summary")
    pdf.key_value("Filename", dataset_info.get("filename", "N/A"))
    pdf.key_value("Rows", str(dataset_info.get("num_rows", "N/A")))
    pdf.key_value("Columns", str(dataset_info.get("num_cols", "N/A")))
    pdf.key_value("Target Column", dataset_info.get("target_column", "N/A"))
    pdf.key_value("Problem Type", problem_labels.get(problem_type, problem_type))
    pdf.ln(3)

    # ── 2. Preprocessing Steps ──
    pdf.section_title("2. Preprocessing Pipeline")
    for i, step in enumerate(preprocessing_steps, 1):
        pdf.body_text(f"  {i}. {step}")

    # ── 3. Model Comparison ──
    pdf.section_title("3. Model Comparison")

    # Table header
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(79, 70, 229)
    pdf.set_text_color(255, 255, 255)

    col_widths = [55, 25]
    headers = ["Model"]
    headers.append("Time (s)")

    if problem_type in ("binary", "multiclass"):
        metric_cols = ["Accuracy", "Precision", "Recall", "F1 Score"]
    else:
        metric_cols = ["MAE", "MSE", "RMSE", "R²"]
    col_widths.extend([27] * len(metric_cols))
    headers.extend(metric_cols)

    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 8, header, border=1, fill=True, align="C")
    pdf.ln()

    # Table rows
    pdf.set_font("Helvetica", "", 9)
    for result in model_results:
        is_best = result.get("is_best", False)
        if is_best:
            pdf.set_fill_color(236, 252, 203)  # Light green highlight
        else:
            pdf.set_fill_color(255, 255, 255)

        pdf.set_text_color(30, 41, 59)
        name = result["model_name"]
        if is_best:
            name += " ★"

        pdf.cell(col_widths[0], 7, name, border=1, fill=True)
        pdf.cell(
            col_widths[1], 7, str(result.get("training_time", "")), border=1, fill=True, align="C"
        )

        metrics = result.get("metrics", {})
        if problem_type in ("binary", "multiclass"):
            for key in ["accuracy", "precision", "recall", "f1_score"]:
                val = metrics.get(key, "N/A")
                pdf.cell(27, 7, str(val), border=1, fill=True, align="C")
        else:
            for key in ["mae", "mse", "rmse", "r2_score"]:
                val = metrics.get(key, "N/A")
                pdf.cell(27, 7, str(val), border=1, fill=True, align="C")
        pdf.ln()

    # ── 4. Best Model Details ──
    best_model = next((r for r in model_results if r.get("is_best")), None)
    if best_model:
        pdf.section_title("4. Best Model Details")
        pdf.key_value("Model", best_model["model_name"])
        pdf.key_value("Training Time", f"{best_model.get('training_time', 'N/A')}s")

        if best_model.get("best_params"):
            params_str = ", ".join(f"{k}={v}" for k, v in best_model["best_params"].items())
            pdf.key_value("Best Parameters", params_str)

        if best_model.get("cv_scores"):
            for metric_name, scores in best_model["cv_scores"].items():
                if isinstance(scores, dict):
                    pdf.key_value(
                        f"CV {metric_name}",
                        f"{scores.get('mean', 'N/A')} ± {scores.get('std', 'N/A')}",
                    )

    # ── 5. Feature Selection ──
    if feature_selection:
        pdf.section_title("5. Feature Selection Results")

        if "random_forest_importance" in feature_selection and isinstance(
            feature_selection["random_forest_importance"], list
        ):
            pdf.body_text("Top Features by Random Forest Importance:")
            for i, feat in enumerate(feature_selection["random_forest_importance"][:10], 1):
                pdf.body_text(f"  {i}. {feat['feature']}: {feat['importance']}")

        if "mutual_information" in feature_selection and isinstance(
            feature_selection["mutual_information"], list
        ):
            pdf.body_text("\nTop Features by Mutual Information:")
            for i, feat in enumerate(feature_selection["mutual_information"][:10], 1):
                pdf.body_text(f"  {i}. {feat['feature']}: {feat['mi_score']}")

    # ── 6. Visualizations ──
    if plots:
        pdf.section_title("6. Visualizations")
        temp_files = []

        for plot_name, b64_data in plots.items():
            if b64_data is None:
                continue
            try:
                img_data = base64.b64decode(b64_data)
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                tmp.write(img_data)
                tmp.close()
                temp_files.append(tmp.name)

                readable_name = plot_name.replace("_", " ").title()
                pdf.body_text(readable_name)

                # Check if we need a new page
                if pdf.get_y() > 180:
                    pdf.add_page()

                pdf.image(tmp.name, x=15, w=180)
                pdf.ln(5)
            except Exception:
                continue

        # Clean up temp files
        for tf in temp_files:
            try:
                os.unlink(tf)
            except Exception:
                pass

    # ── 7. Conclusion ──
    pdf.section_title("7. Conclusion")
    if best_model:
        pdf.body_text(
            f"After training and evaluating all models, the best performing model is "
            f"{best_model['model_name']} with the following key metrics:"
        )
        metrics = best_model.get("metrics", {})
        if problem_type in ("binary", "multiclass"):
            pdf.body_text(f"  - Accuracy: {metrics.get('accuracy', 'N/A')}")
            pdf.body_text(f"  - F1 Score: {metrics.get('f1_score', 'N/A')}")
            if metrics.get("roc_auc"):
                pdf.body_text(f"  - ROC-AUC: {metrics.get('roc_auc', 'N/A')}")
        else:
            pdf.body_text(f"  - R² Score: {metrics.get('r2_score', 'N/A')}")
            pdf.body_text(f"  - RMSE: {metrics.get('rmse', 'N/A')}")

    pdf.output(filepath)
    return filepath
