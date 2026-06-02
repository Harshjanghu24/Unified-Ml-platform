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

### Predictions
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Perform a single prediction. Optionally includes SHAP explanations. |
| `POST` | `/predict-batch` | Perform predictions on an uploaded CSV file. |

### Model Management
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/models` | List all saved models and their metadata. |
| `DELETE`| `/model/{id}` | Delete a specific model from the database and disk. |
| `POST` | `/generate-report` | Generate and download a PDF summary report. |

## Schemas
The API uses JSON for request and response bodies. For detailed request/response schemas, refer to the interactive documentation (Swagger UI) at `/docs` when the server is running.
