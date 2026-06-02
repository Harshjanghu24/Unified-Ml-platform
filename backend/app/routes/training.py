"""
Training API Routes.
Handles model training, preprocessing, feature selection,
hyperparameter tuning, and cross-validation orchestration.
"""

import os
import uuid

import joblib
import psutil
from fastapi import APIRouter, BackgroundTasks, HTTPException

from ..database import save_model_record
from ..ml.eda import (
    generate_actual_vs_predicted,
    generate_all_eda,
    generate_confusion_matrix_plot,
    generate_residual_plot,
    generate_roc_curve,
)
from ..ml.explainability import generate_shap_explanations
from ..ml.feature_selection import run_feature_selection
from ..ml.preprocessor import build_preprocessing_pipeline
from ..ml.trainer import train_models
from ..routes.dataset import get_current_dataset

router = APIRouter(prefix="/api", tags=["Training"])

# In-memory store for training results and background jobs
_training_state = {}
_jobs = {}


def get_training_state():
    """Access training state from other modules."""
    return _training_state


MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def _run_training_pipeline_task(job_id: str, dataset: dict):
    """Background task to run the ML pipeline."""
    try:
        df = dataset["df"]
        target_column = dataset["target_column"]
        problem_type = dataset["problem_type"]
        dataset_id = dataset["dataset_id"]

        _jobs[job_id]["progress"] = "Step 1: Preprocessing data..."
        prep_result = build_preprocessing_pipeline(df, target_column, problem_type)

        X_train = prep_result["X_train"]
        X_test = prep_result["X_test"]
        y_train = prep_result["y_train"]
        y_test = prep_result["y_test"]
        feature_names = prep_result["feature_names"]
        scaler = prep_result["scaler"]
        label_encoder = prep_result["label_encoder"]
        steps_log = prep_result["steps_log"]

        _jobs[job_id]["progress"] = "Step 2: Feature Selection..."
        try:
            feature_selection = run_feature_selection(
                X_train, y_train, feature_names, problem_type, top_n=15
            )
        except Exception as e:
            feature_selection = {"error": str(e)}

        _jobs[job_id]["progress"] = "Step 3: Training Models (this may take a while)..."
        model_results = train_models(X_train, X_test, y_train, y_test, problem_type)

        _jobs[job_id]["progress"] = "Step 4: Generating Evaluation Plots..."
        eval_plots = {}
        best_model_result = next((r for r in model_results if r["is_best"]), model_results[0])
        best_model = best_model_result["model_object"]

        if problem_type in ("binary", "multiclass"):
            cm = best_model_result["metrics"].get("confusion_matrix")
            if cm:
                classes = None
                if label_encoder:
                    classes = [str(c) for c in label_encoder.classes_]
                eval_plots["confusion_matrix"] = generate_confusion_matrix_plot(cm, classes)

            n_classes = int(y_test.nunique())
            roc = generate_roc_curve(best_model, X_test, y_test, n_classes)
            if roc:
                eval_plots["roc_curve"] = roc
        else:
            y_pred = best_model_result["metrics"].get("y_pred", [])
            y_actual = best_model_result["metrics"].get("y_test", y_test.tolist())
            if y_pred:
                eval_plots["actual_vs_predicted"] = generate_actual_vs_predicted(y_actual, y_pred)
                eval_plots["residual_plot"] = generate_residual_plot(y_actual, y_pred)

        _jobs[job_id]["progress"] = "Step 5: Generating SHAP Explainability..."
        try:
            shap_results = generate_shap_explanations(
                best_model, X_train, X_test, feature_names, problem_type
            )
        except Exception as e:
            shap_results = {"error": str(e)}

        _jobs[job_id]["progress"] = "Step 6: Generating EDA Plots..."
        try:
            eda_plots = generate_all_eda(df, target_column, problem_type)
        except Exception as e:
            eda_plots = {"error": str(e)}

        _jobs[job_id]["progress"] = "Step 7: Saving results..."
        saved_models = []
        for result in model_results:
            model_filename = f"{result['model_name']}_{dataset_id}.joblib"
            model_path = os.path.join(MODELS_DIR, model_filename)
            joblib.dump(result["model_object"], model_path)

            clean_metrics = {
                k: v for k, v in result["metrics"].items() if k not in ("y_pred", "y_test")
            }

            model_id = save_model_record(
                dataset_id=dataset_id,
                model_name=result["model_name"],
                problem_type=problem_type,
                metrics=clean_metrics,
                training_time=result["training_time"],
                model_path=model_path,
                is_best=result["is_best"],
                best_params=result.get("best_params"),
                cv_scores=result.get("cv_scores"),
            )

            saved_models.append(
                {
                    "model_id": model_id,
                    "model_name": result["model_name"],
                    "is_best": result["is_best"],
                    "metrics": clean_metrics,
                    "training_time": result["training_time"],
                    "best_params": result.get("best_params", {}),
                    "best_search_score": result.get("best_search_score"),
                    "cv_scores": result.get("cv_scores", {}),
                }
            )

        pipeline_path = os.path.join(MODELS_DIR, f"pipeline_{dataset_id}.joblib")
        joblib.dump(
            {
                "scaler": scaler,
                "label_encoder": label_encoder,
                "feature_names": feature_names,
                "problem_type": problem_type,
                "target_column": target_column,
            },
            pipeline_path,
        )

        _training_state["models"] = saved_models
        _training_state["preprocessing_steps"] = steps_log
        _training_state["feature_selection"] = feature_selection
        _training_state["eval_plots"] = eval_plots
        _training_state["eda_plots"] = eda_plots
        _training_state["shap_results"] = shap_results
        _training_state["problem_type"] = problem_type
        _training_state["dataset_id"] = dataset_id
        _training_state["best_model_name"] = best_model_result["model_name"]
        _training_state["pipeline_path"] = pipeline_path

        _jobs[job_id]["status"] = "completed"
        _jobs[job_id]["progress"] = "Training completed successfully."
        _jobs[job_id]["result"] = {
            "message": "Training completed successfully",
            "problem_type": problem_type,
            "models": saved_models,
            "preprocessing_steps": steps_log,
            "feature_selection": feature_selection,
            "best_model": best_model_result["model_name"],
        }
    except Exception as e:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)


@router.post("/train")
async def train(background_tasks: BackgroundTasks):
    dataset = get_current_dataset()
    if dataset.get("df") is None:
        raise HTTPException(
            status_code=400, detail="No dataset uploaded. Please upload a dataset first."
        )

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status": "running",
        "progress": "Initializing training pipeline...",
        "result": None,
        "error": None,
    }
    background_tasks.add_task(_run_training_pipeline_task, job_id, dataset)
    return {"job_id": job_id, "message": "Training started in the background."}


@router.get("/train-status/{job_id}")
async def get_train_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = _jobs[job_id]
    if job["status"] == "failed":
        return {"status": "failed", "error": job["error"]}

    return {"status": job["status"], "progress": job["progress"]}


@router.get("/train-result/{job_id}")
async def get_train_result(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = _jobs[job_id]
    if job["status"] == "running":
        raise HTTPException(status_code=400, detail="Job is still running")
    elif job["status"] == "failed":
        raise HTTPException(status_code=500, detail=f"Job failed: {job['error']}")

    return job["result"]


@router.get("/metrics")
async def get_metrics():
    if not _training_state.get("models"):
        raise HTTPException(status_code=404, detail="No models trained yet. Run /api/train first.")

    return {
        "models": _training_state["models"],
        "preprocessing_steps": _training_state.get("preprocessing_steps", []),
        "feature_selection": _training_state.get("feature_selection", {}),
        "eval_plots": _training_state.get("eval_plots", {}),
        "shap_results": {
            k: v
            for k, v in _training_state.get("shap_results", {}).items()
            if k != "error" or v is not None
        },
        "problem_type": _training_state.get("problem_type"),
        "best_model": _training_state.get("best_model_name"),
    }


@router.get("/eda")
async def get_eda():
    if not _training_state.get("eda_plots"):
        dataset = get_current_dataset()
        if dataset.get("df") is not None:
            try:
                eda_plots = generate_all_eda(
                    dataset["df"], dataset["target_column"], dataset["problem_type"]
                )
                return {"plots": eda_plots}
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"EDA generation failed: {str(e)}"
                ) from e
        raise HTTPException(status_code=404, detail="No dataset or training data available.")

    return {"plots": _training_state["eda_plots"]}


@router.get("/system-info")
async def get_system_info():
    """Return CPU, RAM, GPU, and CUDA status."""
    info = {
        "cpu": f"{psutil.cpu_percent()}% usage ({psutil.cpu_count(logical=True)} cores)",
        "ram": (
            f"{round(psutil.virtual_memory().used / (1024**3), 2)}GB / "
            f"{round(psutil.virtual_memory().total / (1024**3), 2)}GB "
            f"({psutil.virtual_memory().percent}%)"
        ),
        "gpu": "Unavailable",
        "cuda_status": False,
        "vram": "N/A",
    }

    # Detect GPU via nvidia-smi (no torch dependency required)
    try:
        import subprocess

        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(", ")
            info["cuda_status"] = True
            info["gpu"] = parts[0]
            total_vram = float(parts[1]) / 1024  # Convert MB to GB
            used_vram = float(parts[2]) / 1024
            info["vram"] = f"{round(used_vram, 2)}GB / {round(total_vram, 2)}GB"
    except Exception:
        pass

    return info
