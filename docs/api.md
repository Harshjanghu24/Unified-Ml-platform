# API Documentation

The Unified ML Platform provides a RESTful API for managing datasets, training models, and making predictions.

## Base URL
All API endpoints are prefixed with `/api`.

## Endpoints

### Dataset Management
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload-dataset` | Upload a CSV file for analysis and training. |
| `POST` | `/analyze-columns` | Perform statistical analysis on dataset columns. |
| `GET`  | `/target-recommendations` | Get ranked recommendations for the target column. |
| `POST` | `/select-target` | Set the target column and detect the problem type. |
| `GET`  | `/dataset-summary` | Retrieve a summary and preview of the current dataset. |
| `GET`  | `/dataset-columns` | List all column names in the current dataset. |

### Model Training
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/train` | Start the asynchronous training pipeline. Returns a `job_id`. |
| `GET`  | `/train-status/{job_id}` | Poll the status and progress of a training job. |
| `GET`  | `/train-result/{job_id}` | Retrieve results once a training job is completed. |
| `GET`  | `/metrics` | Get evaluation metrics for all trained models. |
| `GET`  | `/eda` | Retrieve generated Exploratory Data Analysis plots. |
| `GET`  | `/system-info` | Get hardware status (CPU, RAM, GPU/CUDA). |

**Example: Start Training**
`POST /api/train`
```json
{
  "target_column": "target",
  "problem_type": "binary_classification",
  "models": ["xgboost", "random_forest"],
  "optimize": true
}
```
*Response (202 Accepted):*
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "message": "Training pipeline started successfully."
}
```

### Predictions
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Perform a single prediction. Optionally includes SHAP explanations. |
| `POST` | `/predict-batch` | Perform predictions on an uploaded CSV file. |

**Example: Single Prediction**
`POST /api/predict`
```json
{
  "model_id": "model_20231027_1230",
  "data": {
    "age": 35,
    "income": 75000,
    "education": "Masters"
  },
  "explain": true
}
```
*Response (200 OK):*
```json
{
  "prediction": 1,
  "probability": 0.85,
  "explanation": "base64_encoded_shap_plot_string..."
}
```

### Model Management
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/models` | List all saved models and their metadata. |
| `DELETE`| `/model/{id}` | Delete a specific model from the database and disk. |
| `POST` | `/generate-report` | Generate and download a PDF summary report. |

## Schemas
The API uses JSON for request and response bodies. For detailed request/response schemas, refer to the interactive documentation (Swagger UI) at `/docs` when the server is running.
