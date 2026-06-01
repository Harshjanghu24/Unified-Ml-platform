# 📋 Project Plan — Unified Supervised Learning AutoML Platform

> **Version:** 1.0.0 | **Author:** Harsh Janghu | **Date:** June 2026  
> **Repository:** [github.com/Harshjanghu24/Unified-Ml-platform](https://github.com/Harshjanghu24/Unified-Ml-platform)

---

## 1. Project Overview

An end-to-end, production-ready AutoML platform that automates the complete supervised learning pipeline — from raw CSV upload to model predictions, explainability, and PDF reports. Supports **Binary Classification**, **Multi-Class Classification**, and **Regression** with automatic problem type detection.

---

## 2. Problem Statement

Building ML models typically requires manual exploration, expert knowledge for problem type selection, repetitive preprocessing, trial-and-error model selection, custom evaluation code, and separate tools for explainability. This platform **unifies the entire workflow** into a single, automated pipeline accessible via a browser.

---

## 3. Objectives

| # | Objective | Status |
|---|-----------|--------|
| 1 | Automated problem type detection (binary / multi-class / regression) | ✅ |
| 2 | AI-powered target column recommendation engine | ✅ |
| 3 | Automated data preprocessing pipeline | ✅ |
| 4 | Multi-model training with hyperparameter tuning (GridSearch / RandomizedSearch) | ✅ |
| 5 | K-Fold Cross Validation for all models | ✅ |
| 6 | Comprehensive evaluation metrics and visualizations | ✅ |
| 7 | Feature selection (Chi-Square, Mutual Information, RF Importance) | ✅ |
| 8 | SHAP-based model explainability | ✅ |
| 9 | Exploratory Data Analysis (EDA) with auto-generated plots | ✅ |
| 10 | Single & batch prediction with real-time SHAP explanations | ✅ |
| 11 | Downloadable PDF report generation | ✅ |
| 12 | GPU (CUDA) acceleration for XGBoost | ✅ |
| 13 | Asynchronous background training with progress tracking | ✅ |
| 14 | Memory-safe preprocessing with cardinality guards | ✅ |
| 15 | Modern, responsive React frontend | ✅ |

---

## 4. Scope

### In Scope
- Supervised Learning — Binary Classification, Multi-Class Classification, Regression
- Tabular CSV data
- Scikit-Learn + XGBoost model families
- Web-based interface (React + FastAPI)
- Single-user local deployment

### Out of Scope (v1.0)
- Unsupervised learning, deep learning, image/text/time-series data
- Multi-user authentication, cloud deployment
- Real-time streaming, model versioning, A/B testing

---

## 5. Technology Stack

### Backend
| Component | Technology |
|-----------|------------|
| Web Framework | FastAPI |
| ML Engine | Scikit-Learn |
| Gradient Boosting | XGBoost (optional GPU/CUDA) |
| Explainability | SHAP |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Report Generation | FPDF2 |
| Database | SQLite |
| System Monitoring | psutil |
| Serialization | Joblib |

### Frontend
| Component | Technology |
|-----------|------------|
| UI Framework | React 19 |
| Build Tool | Vite 8 |
| Routing | React Router v7 |
| HTTP Client | Axios |
| Styling | TailwindCSS v4 |
| Charts | Recharts |
| Animations | Framer Motion |
| Notifications | React Hot Toast |
| Icons | React Icons |

---

## 6. Development Phases

```
Phase 1: Foundation        → Project setup, FastAPI skeleton, React scaffold
Phase 2: Data Pipeline     → CSV upload, preprocessing, problem detection
Phase 3: Training Engine   → Model training, tuning, cross-validation
Phase 4: Evaluation        → Metrics, plots, confusion matrices, ROC curves
Phase 5: Intelligence      → Feature selection, SHAP explainability
Phase 6: User Experience   → Prediction UI, EDA page, report generation
Phase 7: Optimization      → Async training, GPU support, memory guards
Phase 8: Stabilization     → Bug fixes, label encoding, edge cases
Phase 9: Documentation     → README, PLAN.md, ARCHITECTURE.md
```

---

## 7. Supported ML Models

### Classification
| Model | Problem Type | Hyperparameters Tuned |
|-------|--------------|-----------------------|
| Logistic Regression | Binary | C, penalty, solver, max_iter |
| Random Forest Classifier | Binary, Multi-Class | n_estimators, max_depth, min_samples_split |
| Decision Tree Classifier | Multi-Class | max_depth, min_samples_split, criterion |
| K-Nearest Neighbors | Multi-Class | n_neighbors, weights, metric |
| XGBoost Classifier | Binary, Multi-Class | n_estimators, max_depth, learning_rate |

### Regression
| Model | Hyperparameters Tuned |
|-------|-----------------------|
| Linear Regression | None (closed-form) |
| Random Forest Regressor | n_estimators, max_depth, min_samples_split |
| XGBoost Regressor | n_estimators, max_depth, learning_rate |

---

## 8. Evaluation Metrics

### Classification
- Accuracy, Precision, Recall, F1 Score, ROC-AUC
- Confusion Matrix, Classification Report
- K-Fold Cross Validation (accuracy, f1_weighted)

### Regression
- MAE, MSE, RMSE, R² Score
- K-Fold Cross Validation (R², neg_MAE)

---

## 9. Risk Analysis

| Risk | Impact | Mitigation |
|------|--------|------------|
| Large datasets cause memory overflow | High | Cardinality limits, feature cap at 500, chunked upload |
| Long training blocks server | High | Async background tasks with progress polling |
| Non-numeric target labels crash metrics | High | Automated LabelEncoder before training |
| XGBoost not installed | Medium | Graceful fallback — excluded if unavailable |
| GPU not available | Low | Auto-detection via nvidia-smi; CPU fallback |
| Non-UTF-8 CSV files | Medium | Multi-encoding fallback chain |

---

## 10. Future Roadmap (v2.0)

- 🔐 User Authentication (JWT)
- ☁️ Docker + Cloud Deployment
- 🧠 Deep Learning (TensorFlow/PyTorch)
- 🔄 Model Versioning & Experiment Tracking
- 📡 REST Prediction API Endpoints
- 📈 Ensemble Methods (Stacking, Voting)
- 🗃️ PostgreSQL for Multi-User Support

---

## 11. Deployment

```bash
# Backend
cd backend && pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend
cd frontend && npm install && npm run dev
```

- **Frontend:** http://localhost:5173
- **Backend API:** http://localhost:8000
- **Swagger Docs:** http://localhost:8000/docs

---

*Last updated: June 2026*
