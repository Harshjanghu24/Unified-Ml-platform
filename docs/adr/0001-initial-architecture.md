# ADR 1: Initial Architecture Choice

## Status
Accepted

## Context
We need to build a Unified ML Platform that supports various problem types (Binary, Multi-Class, Regression) with a focus on ease of use, performance, and explainability. The platform should handle data ingestion, automated analysis, model training, evaluation, and inference.

## Decision
We chose the following stack for the initial implementation:

### 1. Backend: FastAPI (Python)
- **Rationale:** 
    - **Performance:** FastAPI is one of the fastest Python frameworks available.
    - **Type Safety:** Leveraging Pydantic for request/response validation.
    - **Developer Productivity:** Automatic OpenAPI (Swagger) documentation.
    - **ML Ecosystem:** Python is the de-facto language for machine learning, providing direct access to Scikit-Learn, XGBoost, and SHAP.
    - **Asynchronous Support:** Essential for handling long-running training jobs without blocking the main event loop.

### 2. Frontend: React (Vite)
- **Rationale:**
    - **Component-Based:** Allows for a modular and maintainable UI.
    - **Ecosystem:** Rich set of libraries for data visualization and state management.
    - **Performance:** Vite provides a lightning-fast development experience and optimized production builds.
    - **Familiarity:** Widely adopted industry standard with a large talent pool.

### 3. Database: SQLite
- **Rationale:**
    - **Zero-Configuration:** No external database server required.
    - **Simplicity:** Perfect for single-user or small-scale internal tools.
    - **Persistence:** Provides a reliable way to track jobs, models, and dataset metadata.

## Consequences
- **Positive:**
    - Fast development cycles.
    - Robust and type-safe API.
    - Seamless integration with existing ML libraries.
- **Negative:**
    - SQLite may not scale well for highly concurrent multi-user environments (not a primary concern for this phase).
    - Managing asynchronous tasks directly in FastAPI (without Celery/Redis) is sufficient for now but might need upgrading as complexity grows.
