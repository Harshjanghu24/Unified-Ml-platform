# 🤖 Unified Supervised Learning AutoML Platform

A production-ready, full-stack **AutoML web application** that intelligently analyzes uploaded datasets, recommends target columns using AI-powered scoring, automatically detects the supervised learning problem type, trains multiple ML models with hyperparameter tuning, compares performance, explains predictions with SHAP, and provides an interactive web interface.

---

## 🌟 Key Features

### 🧠 AI-Powered Target Recommendation Engine
Unlike traditional AutoML tools, this platform **does NOT require users to manually specify the target column**. Instead:
- Upload any labeled CSV dataset
- The AI Analyzer examines **every column**, computing Shannon Entropy, Variance, Cardinality, Distribution Balance, and Data Type
- A **ranked recommendation table** displays:
  - Suggested Problem Type (Binary / Multi-Class / Regression)
  - Confidence Score (0–100%)
  - Recommendation Rating (Excellent / Good / Fair / Poor / Unsuitable)
- User selects from the AI-recommended targets

### 🎯 Three Problem Types
| Type | Models | Metrics |
|------|--------|---------|
| **Binary Classification** | Logistic Regression, Random Forest, XGBoost | Accuracy, Precision, Recall, F1, ROC-AUC |
| **Multi-Class Classification** | Decision Tree, Random Forest, KNN, XGBoost | Accuracy, Precision, Recall, F1 (weighted) |
| **Regression** | Linear Regression, Random Forest, XGBoost | MAE, MSE, RMSE, R² |

### ⚙️ Complete ML Pipeline
- **Preprocessing**: Missing value imputation, one-hot encoding, feature scaling, stratified train-test split
- **Feature Selection**: Chi-Square test, Mutual Information, Random Forest importance
- **Hyperparameter Tuning**: GridSearchCV / RandomizedSearchCV
- **Cross-Validation**: K-Fold (k=5)
- **Explainability**: SHAP summary & bar plots, per-prediction explanations
- **Reporting**: Auto-generated downloadable PDF reports

---

## 📊 AutoML Workflow

```
┌──────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│  1. Upload   │───▶│  2. AI Analyzes  │───▶│  3. Recommendations │
│    CSV       │    │   All Columns    │    │    (Ranked Table)   │
└──────────────┘    └──────────────────┘    └─────────┬───────────┘
                                                      │
┌──────────────┐    ┌──────────────────┐    ┌─────────▼───────────┐
│  6. Evaluate │◀───│  5. Train Models │◀───│  4. Select Target   │
│  & Compare   │    │  + GridSearchCV  │    │     Column          │
└──────┬───────┘    └──────────────────┘    └─────────────────────┘
       │
┌──────▼───────┐    ┌──────────────────┐
│  7. SHAP     │───▶│  8. Predict      │
│  Explain     │    │  (Single/Batch)  │
└──────────────┘    └──────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.12, FastAPI, Uvicorn |
| **ML/AI** | Scikit-Learn, XGBoost, SHAP |
| **Data Processing** | Pandas, NumPy, SciPy |
| **Visualization** | Matplotlib, Seaborn |
| **Database** | SQLite |
| **Frontend** | React 19, Vite |
| **Styling** | Tailwind CSS v4, Framer Motion |
| **Charts** | Recharts |
| **HTTP Client** | Axios |
| **Report** | fpdf2 (PDF generation) |

---

## 🏗️ Project Structure

```
unified-ml-platform/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI entry point
│   │   ├── database.py              # SQLite database operations
│   │   ├── ml/
│   │   │   ├── analyzer.py          # AI Target Recommendation Engine
│   │   │   ├── detector.py          # Problem type detector
│   │   │   ├── preprocessor.py      # Preprocessing pipeline
│   │   │   ├── trainer.py           # Model training (+ XGBoost)
│   │   │   ├── feature_selection.py # Chi-Square, MI, RF importance
│   │   │   ├── explainability.py    # SHAP explanations
│   │   │   ├── eda.py               # EDA visualizations
│   │   │   └── report_generator.py  # PDF report generation
│   │   └── routes/
│   │       ├── dataset.py           # Upload, analyze, recommend, select
│   │       ├── training.py          # Training pipeline endpoint
│   │       ├── prediction.py        # Single & batch predictions
│   │       └── models.py            # Model management & reports
│   ├── data/                        # SQLite DB (auto-created)
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx                  # Root component with routing
│   │   ├── main.jsx                 # Entry point
│   │   ├── index.css                # Design system (dark/light)
│   │   ├── components/
│   │   │   └── Sidebar.jsx          # Navigation sidebar
│   │   ├── pages/
│   │   │   ├── HomePage.jsx         # Landing page
│   │   │   ├── DatasetPage.jsx      # CSV upload (no target needed)
│   │   │   ├── AnalyzerPage.jsx     # AI target recommendations
│   │   │   ├── EDAPage.jsx          # EDA visualizations
│   │   │   ├── TrainingPage.jsx     # Model training
│   │   │   ├── EvaluationPage.jsx   # Model comparison
│   │   │   ├── ExplainabilityPage.jsx # SHAP dashboard
│   │   │   ├── PredictionPage.jsx   # Predictions
│   │   │   └── SavedModelsPage.jsx  # Model management
│   │   └── services/
│   │       └── api.js               # Axios API client
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── README.md
└── .gitignore
```

---

## 📖 Documentation

For detailed information on the platform's design and operation, please refer to the following guides:

- **[Architecture Guide](docs/architecture.md)**: Deep dive into the system components and data flow.
- **[API Reference](docs/api.md)**: Detailed documentation for all REST API endpoints.
- **[Deployment Guide](docs/deployment.md)**: Instructions for setting up and deploying the platform.
- **[Troubleshooting Guide](docs/troubleshooting.md)**: Common issues and their solutions.
- **[Architectural Decision Records (ADR)](docs/adr/)**: History of key technical decisions.

---

## 🚀 Installation & Setup

### Prerequisites
- **Python** 3.12.4
- **Node.js** 18 or higher
- **pip** (Python package manager)
- **npm** (Node package manager)

### Backend Setup

```bash
# Navigate to backend
cd backend

# (Optional) Create virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Start the FastAPI backend server
uvicorn app.main:app --reload --reload-exclude "app/data/*" --host 0.0.0.0 --port 8000
```

The backend will run at **http://localhost:8000**  
API documentation is available at **http://localhost:8000/docs**

### Frontend Setup

```bash
# Navigate to frontend
cd frontend

# Install Node dependencies
npm install

# Start the Vite development server
npm run dev
```

The frontend will run at **http://localhost:5173** and automatically proxies API requests to the backend.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload-dataset` | Upload CSV file (no target needed) |
| `POST` | `/api/analyze-columns` | Analyze all columns for target suitability |
| `GET` | `/api/target-recommendations` | Get ranked target recommendations |
| `POST` | `/api/select-target` | Select a target column |
| `GET` | `/api/dataset-summary` | Current dataset summary & preview |
| `POST` | `/api/train` | Run full ML training pipeline |
| `GET` | `/api/metrics` | Retrieve training results & SHAP data |
| `GET` | `/api/eda` | Generate EDA visualizations |
| `POST` | `/api/predict` | Single-row prediction (with SHAP) |
| `POST` | `/api/predict-batch` | Batch CSV prediction |
| `GET` | `/api/models` | List all saved models |
| `DELETE` | `/api/model/{id}` | Delete a saved model |
| `POST` | `/api/generate-report` | Download PDF analysis report |

---

## 📸 Screenshots

> _Screenshots of the running platform_

### Home Page
The landing page showcasing the platform's capabilities and the 8-step ML pipeline.

### Dataset Analyzer
The AI-powered target recommendation engine displaying confidence scores for every column.

### Training Dashboard
One-click training with hyperparameter tuning, cross-validation, and model comparison.

### Explainability Dashboard
SHAP-based feature importance rankings and beeswarm plots.

### Prediction Page
Single and batch prediction modes with confidence scores and SHAP explanations.

---

## 🎨 UI Features

- 🌙 **Dark Mode** + ☀️ **Light Mode** toggle
- 📱 **Responsive Design** with collapsible sidebar
- 📊 **Interactive Charts** (Recharts)
- ✨ **Smooth Animations** (Framer Motion)
- 🔔 **Toast Notifications** for all actions
- 📈 **Progress Indicators** during training
- 🏆 **Best Model** badge in comparison dashboard

---

## 🔮 Future Improvements

- [ ] Deep Learning models (Neural Networks via TensorFlow/PyTorch)
- [ ] Time-series forecasting support
- [ ] Feature engineering suggestions
- [ ] Model deployment (Docker + REST API export)
- [ ] Multi-user support with authentication
- [ ] Dataset versioning and experiment tracking
- [ ] Automated feature engineering (polynomial, interaction features)
- [ ] Support for Excel, JSON, and Parquet file formats
- [ ] Real-time training progress via WebSocket
- [ ] Model ensembling (stacking, blending)

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---

<p align="center">
  Built with ❤️ using FastAPI, React, Scikit-Learn, XGBoost & SHAP
</p>
