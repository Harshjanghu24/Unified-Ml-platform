# 🏗️ Architecture — Unified Supervised Learning AutoML Platform

> **Version:** 1.0.0 | **Date:** June 2026  
> **Repository:** [github.com/Harshjanghu24/Unified-Ml-platform](https://github.com/Harshjanghu24/Unified-Ml-platform)

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          FRONTEND (React 19 + Vite 8)              │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐          │
│  │ Dataset  │ Analyzer │ Training │ Evaluate │ Predict  │          │
│  │  Page    │   Page   │   Page   │   Page   │   Page   │          │
│  └────┬─────┴────┬─────┴────┬─────┴────┬─────┴────┬─────┘          │
│       │          │          │          │          │                  │
│       └──────────┴──────┬───┴──────────┴──────────┘                  │
│                         │  Axios API Service Layer                   │
├─────────────────────────┼───────────────────────────────────────────┤
│                    HTTP REST (JSON)                                  │
├─────────────────────────┼───────────────────────────────────────────┤
│                         │                                           │
│              BACKEND (FastAPI + Uvicorn)                             │
│  ┌──────────────────────┴──────────────────────────┐                │
│  │              API Router Layer                    │                │
│  │  ┌──────────┬──────────┬──────────┬──────────┐  │                │
│  │  │ dataset  │ training │prediction│  models  │  │                │
│  │  │  .py     │   .py    │   .py    │   .py    │  │                │
│  │  └────┬─────┴────┬─────┴────┬─────┴────┬─────┘  │                │
│  ├───────┼──────────┼──────────┼──────────┼────────┤                │
│  │       │     ML Engine Layer (app/ml/)   │        │                │
│  │  ┌────┴────┬─────┴────┬─────┴────┬─────┴────┐   │                │
│  │  │analyzer │preproces │ trainer  │explaina  │   │                │
│  │  │detector │  sor     │  eda     │  bility  │   │                │
│  │  │         │feature_  │         │report_   │   │                │
│  │  │         │selection │         │generator │   │                │
│  │  └─────────┴──────────┴─────────┴──────────┘   │                │
│  ├─────────────────────────────────────────────────┤                │
│  │            Persistence Layer                     │                │
│  │  ┌──────────────┐  ┌────────────────────────┐   │                │
│  │  │  SQLite DB    │  │  Joblib Model Files    │   │                │
│  │  │  (platform.db)│  │  (data/models/*.joblib)│   │                │
│  │  └──────────────┘  └────────────────────────┘   │                │
│  └─────────────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Directory Structure

```
unified-ml-platform/
├── backend/
│   ├── app/
│   │   ├── __init__.py              # Package marker
│   │   ├── main.py                  # FastAPI app entry point, CORS, lifespan
│   │   ├── database.py              # SQLite connection, schema, CRUD helpers
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── dataset.py           # Upload, analyze, recommend, select target
│   │   │   ├── training.py          # Train, status polling, metrics, EDA, system info
│   │   │   ├── prediction.py        # Single & batch prediction with SHAP
│   │   │   └── models.py            # List, delete, report generation
│   │   ├── ml/
│   │   │   ├── __init__.py
│   │   │   ├── analyzer.py          # AI target column recommendation engine
│   │   │   ├── detector.py          # Problem type auto-detection
│   │   │   ├── preprocessor.py      # Imputation, encoding, scaling, split
│   │   │   ├── trainer.py           # Multi-model training + hyperparameter tuning
│   │   │   ├── feature_selection.py # Chi-Square, MI, RF importance
│   │   │   ├── eda.py               # Plot generation (base64 PNGs)
│   │   │   ├── explainability.py    # SHAP summary/bar plots + single explanations
│   │   │   └── report_generator.py  # PDF report auto-generation
│   │   └── data/
│   │       ├── uploads/             # Uploaded CSV files
│   │       ├── models/              # Serialized .joblib model files
│   │       ├── reports/             # Generated PDF reports
│   │       └── platform.db          # SQLite database
│   └── requirements.txt             # Python dependencies
│
├── frontend/
│   ├── index.html                   # HTML entry point
│   ├── vite.config.js               # Vite + React + proxy config
│   ├── package.json                 # Node.js dependencies
│   └── src/
│       ├── main.jsx                 # React DOM render entry
│       ├── App.jsx                  # Root component with routing
│       ├── index.css                # Global styles + theme system
│       ├── components/
│       │   └── Sidebar.jsx          # Collapsible navigation sidebar
│       ├── pages/
│       │   ├── HomePage.jsx         # Landing page + workflow overview
│       │   ├── DatasetPage.jsx      # Dataset summary & preview
│       │   ├── AnalyzerPage.jsx     # Upload + column analysis + target selection
│       │   ├── EDAPage.jsx          # Exploratory Data Analysis plots
│       │   ├── TrainingPage.jsx     # Model training + progress + system info
│       │   ├── EvaluationPage.jsx   # Metrics, plots, model comparison
│       │   ├── ExplainabilityPage.jsx # SHAP plots + feature importance
│       │   ├── PredictionPage.jsx   # Single & batch predictions
│       │   └── SavedModelsPage.jsx  # Model management + report download
│       └── services/
│           └── api.js               # Axios API client (all endpoints)
│
├── README.md
├── PLAN.md
├── ARCHITECTURE.md
└── .gitignore
```

---

## 3. Data Flow Pipeline

```
                    User uploads CSV
                          │
                          ▼
              ┌───────────────────────┐
              │   1. UPLOAD DATASET   │  POST /api/upload-dataset
              │   • Save to disk     │  • Chunked file write
              │   • Read with Pandas │  • Multi-encoding fallback
              │   • Compute summary  │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  2. ANALYZE COLUMNS   │  POST /api/analyze-columns
              │  • Entropy, Variance │  • Cardinality ratio
              │  • Suitability scores│  • Binary/Multi/Regression scores
              │  • Recommendations   │  GET /api/target-recommendations
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  3. SELECT TARGET     │  POST /api/select-target
              │  • Detect problem    │  • Binary / Multiclass / Regression
              │  • Save to DB        │  • Target info & class distribution
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  4. TRAIN MODELS      │  POST /api/train (async)
              │  ┌─────────────────┐  │
              │  │ Preprocessing   │  │  • Missing value imputation
              │  │ Pipeline        │  │  • One-hot encoding
              │  │                 │  │  • Label encoding (target)
              │  │                 │  │  • StandardScaler
              │  │                 │  │  • 80/20 train-test split
              │  └────────┬────────┘  │
              │           │           │
              │  ┌────────▼────────┐  │
              │  │ Feature         │  │  • Chi-Square test
              │  │ Selection       │  │  • Mutual Information
              │  │                 │  │  • Random Forest importance
              │  └────────┬────────┘  │
              │           │           │
              │  ┌────────▼────────┐  │
              │  │ Model Training  │  │  • GridSearchCV / RandomizedSearchCV
              │  │ + Tuning        │  │  • K-Fold Cross Validation
              │  │                 │  │  • GPU detection for XGBoost
              │  └────────┬────────┘  │
              │           │           │
              │  ┌────────▼────────┐  │
              │  │ Evaluation &    │  │  • Metrics computation
              │  │ Visualization   │  │  • Confusion matrix, ROC, residuals
              │  └────────┬────────┘  │
              │           │           │
              │  ┌────────▼────────┐  │
              │  │ SHAP            │  │  • TreeExplainer / LinearExplainer
              │  │ Explainability  │  │  • Summary + bar plots
              │  └────────┬────────┘  │
              │           │           │
              │  ┌────────▼────────┐  │
              │  │ EDA Plots       │  │  • Histograms, box plots
              │  │                 │  │  • Correlation heatmap, scatter
              │  └────────┬────────┘  │
              │           │           │
              │  ┌────────▼────────┐  │
              │  │ Save Results    │  │  • Joblib model serialization
              │  │                 │  │  • SQLite metadata records
              │  └─────────────────┘  │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  5. PREDICT           │  POST /api/predict
              │  • Load best model   │  POST /api/predict-batch
              │  • Align features    │  • Real-time SHAP explanations
              │  • Decode labels     │
              └───────────────────────┘
```

---

## 4. API Endpoints

### Dataset Routes (`/api`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload-dataset` | Upload CSV file (chunked, multi-encoding) |
| POST | `/analyze-columns` | Analyze all columns for target suitability |
| GET | `/target-recommendations` | Get ranked target column recommendations |
| POST | `/select-target` | Select target column, detect problem type |
| GET | `/dataset-summary` | Get current dataset summary + preview |
| GET | `/dataset-columns` | Get column names list |

### Training Routes (`/api`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/train` | Start async training pipeline (returns job_id) |
| GET | `/train-status/{job_id}` | Poll training progress |
| GET | `/train-result/{job_id}` | Get completed training results |
| GET | `/metrics` | Get all model metrics + evaluation data |
| GET | `/eda` | Get EDA visualization plots |
| GET | `/system-info` | Get CPU, RAM, GPU, CUDA status |

### Prediction Routes (`/api`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Single prediction with optional SHAP |
| POST | `/predict-batch` | Batch CSV prediction |

### Model Management Routes (`/api`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/models` | List all saved models |
| DELETE | `/model/{id}` | Delete a model record + file |
| POST | `/generate-report` | Generate downloadable PDF report |

---

## 5. ML Engine Architecture

### 5.1 Problem Type Detection (`detector.py`)

```
Target Column Analysis:
  ├── Non-numeric (object/category)
  │   ├── 2 unique values    → Binary Classification
  │   └── >2 unique values   → Multi-Class Classification
  └── Numeric
      ├── ≤2 unique          → Binary Classification
      ├── 3-20 unique (int)  → Multi-Class Classification
      ├── 3-20 unique (int-like float) → Multi-Class Classification
      └── >20 unique / float → Regression
```

### 5.2 Preprocessing Pipeline (`preprocessor.py`)

```
Step 1: Separate features (X) and target (y)
Step 2: Drop high-cardinality columns (>20 unique categorical, ID columns)
Step 3: Identify numeric vs categorical columns
Step 4: Impute missing values (mean for numeric, most_frequent for categorical)
Step 5: One-Hot Encode categorical features (drop_first=False)
Step 6: Guard against feature explosion (cap at 500 columns)
Step 7: Label Encode target for classification problems
Step 8: StandardScaler on all features
Step 9: Train-Test Split (80/20, stratified when ≤20 classes)
```

### 5.3 Training Engine (`trainer.py`)

```
For each model in the problem-type model set:
  1. Select hyperparameter grid from PARAM_GRIDS
  2. If grid_size ≤ 50 → GridSearchCV (exhaustive)
     If grid_size > 50  → RandomizedSearchCV (20 iterations)
  3. Fit best estimator
  4. Evaluate on test set
  5. K-Fold Cross Validation (cv=5)
  6. Record metrics, timing, best params

Best model = max(f1_score) for classification
Best model = max(r2_score) for regression
```

### 5.4 SHAP Explainability (`explainability.py`)

```
Explainer Selection:
  ├── Forest/Tree models    → TreeExplainer (fast, exact)
  ├── Linear/Logistic       → LinearExplainer (fast, exact)
  └── KNN / other           → KernelExplainer (slow, approximate)

Outputs:
  • Summary plot (beeswarm) — base64 PNG
  • Bar plot (mean |SHAP|) — base64 PNG
  • Feature importance ranking — JSON list
  • Single prediction explanations — top 10 contributors
```

---

## 6. Database Schema

```sql
-- datasets table
CREATE TABLE datasets (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    filename        TEXT NOT NULL,
    target_column   TEXT,
    problem_type    TEXT,           -- 'binary' | 'multiclass' | 'regression'
    num_rows        INTEGER,
    num_cols        INTEGER,
    upload_time     TEXT,           -- ISO 8601
    summary         TEXT            -- JSON blob
);

-- trained_models table
CREATE TABLE trained_models (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id      INTEGER,        -- FK → datasets.id
    model_name      TEXT NOT NULL,
    problem_type    TEXT,
    metrics         TEXT,           -- JSON blob
    training_time   REAL,
    model_path      TEXT,           -- filesystem path to .joblib
    is_best         INTEGER DEFAULT 0,
    best_params     TEXT,           -- JSON blob
    cv_scores       TEXT,           -- JSON blob
    created_at      TEXT,           -- ISO 8601
    FOREIGN KEY (dataset_id) REFERENCES datasets(id)
);
```

---

## 7. Frontend Architecture

### 7.1 Page Routing

| Route | Page Component | Purpose |
|-------|----------------|---------|
| `/` | HomePage | Landing page, workflow overview |
| `/dataset` | DatasetPage | Dataset summary & data preview |
| `/analyzer` | AnalyzerPage | Upload CSV, analyze columns, select target |
| `/eda` | EDAPage | Histograms, box plots, heatmap, scatter |
| `/training` | TrainingPage | Start training, progress bar, system info |
| `/evaluation` | EvaluationPage | Model comparison, metrics, plots |
| `/explainability` | ExplainabilityPage | SHAP plots, feature importance |
| `/prediction` | PredictionPage | Single & batch prediction forms |
| `/models` | SavedModelsPage | Model list, delete, PDF report |

### 7.2 State Management
- **No global store** (Redux/Zustand) — each page fetches data from the API
- **In-memory backend state** (`_current_dataset`, `_training_state`) acts as the session store
- **Auto-restore** — if backend restarts, dataset is auto-reloaded from disk using the latest DB record

### 7.3 API Proxy
Vite dev server proxies `/api/*` requests to `http://localhost:8000` to avoid CORS during development.

---

## 8. Async Training Architecture

```
Frontend                    Backend
   │                           │
   │  POST /api/train          │
   │ ─────────────────────►    │
   │                           │  Create job_id
   │  { job_id: "abc-123" }    │  Start BackgroundTask
   │ ◄─────────────────────    │
   │                           │
   │  GET /train-status/abc    │  ┌──────────────────┐
   │ ─────────────────────►    │  │ Background Task   │
   │  { status: "running",     │  │ Step 1: Preproc   │
   │    progress: "Step 2..." }│  │ Step 2: Features   │
   │ ◄─────────────────────    │  │ Step 3: Training   │
   │                           │  │ Step 4: Eval plots │
   │  (poll every 2 seconds)   │  │ Step 5: SHAP       │
   │ ─────────────────────►    │  │ Step 6: EDA        │
   │  { status: "completed" }  │  │ Step 7: Save       │
   │ ◄─────────────────────    │  └──────────────────┘
   │                           │
   │  GET /train-result/abc    │
   │ ─────────────────────►    │
   │  { models, metrics, ... } │
   │ ◄─────────────────────    │
```

---

## 9. Memory Safety Design

| Guard | Location | Rule |
|-------|----------|------|
| Chunked file upload | `dataset.py` | 1MB chunks to prevent memory spike |
| SHAP subsampling | `explainability.py` | Max 100 test samples, 200 background samples |

---

## 10. GPU Acceleration

```
XGBoost GPU Detection Flow:
  1. Attempt to instantiate XGBClassifier(device="cuda")
  2. Try fitting on trivial data [[0]], [0]
  3. If success → set device="cuda", tree_method="hist"
  4. If fails   → silently fall back to CPU

System Info Endpoint:
  1. Run `nvidia-smi --query-gpu=name,memory.total,memory.used`
  2. Parse GPU name, VRAM usage
  3. No torch dependency required
```

---

## 11. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **SQLite** over PostgreSQL | Single-user, zero-config, no external services |
| **In-memory dataset state** | Fast access during training; auto-restored from disk |
| **Base64 plot encoding** | No file serving needed; plots embedded directly in JSON |
| **Background tasks** (not Celery) | No Redis/broker dependency; sufficient for single-user |
| **Joblib** for serialization | Native sklearn support, fast binary I/O |
| **No torch dependency** | nvidia-smi for GPU detection avoids heavy PyTorch install |
| **Vite proxy** | Clean API calls in dev; no CORS complexity |
| **LabelEncoder on target** | Ensures numeric labels for all sklearn metric functions |

---

*Last updated: June 2026*
