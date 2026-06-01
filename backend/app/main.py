"""
Unified Supervised Learning Platform — FastAPI Application Entry Point.

This is the main application module that:
  - Initializes the FastAPI app with CORS middleware
  - Creates database tables on startup
  - Includes all API routers
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .database import init_db
from .routes import dataset, training, prediction, models


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    init_db()
    print("[OK] Database initialized")
    print("[READY] Unified Supervised Learning Platform is ready!")
    yield


app = FastAPI(
    title="Unified Supervised Learning Platform",
    description=(
        "An AutoML-style platform for Binary Classification, "
        "Multi-Class Classification, and Regression. "
        "Upload a dataset, auto-detect problem type, train models, "
        "compare performance, and make predictions."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(dataset.router)
app.include_router(training.router)
app.include_router(prediction.router)
app.include_router(models.router)


@app.get("/")
async def root():
    return {
        "name": "Unified Supervised Learning Platform",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }
