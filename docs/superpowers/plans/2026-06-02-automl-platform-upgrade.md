# AutoML Platform Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the Unified ML Platform into a production-grade open-source repository with governance, CI/CD, robust testing, logging, and ML best practices.

**Architecture:** We are preserving the existing FastAPI and React stack. We will centralize configuration using pydantic-settings, add structured logging, configure testing frameworks (pytest, vitest), establish standard repository governance, and configure GitHub Actions for CI/CD and release engineering.

**Tech Stack:** FastAPI, React, Pytest, Vitest, GitHub Actions, Pydantic, MLflow, Pre-commit.

---

### Task 1: Repository Governance

**Files:**
- Create: `CONTRIBUTING.md`
- Create: `SECURITY.md`
- Create: `CODE_OF_CONDUCT.md`
- Create: `CHANGELOG.md`
- Create: `CODEOWNERS`
- Create: `.github/ISSUE_TEMPLATE/bug_report.md`
- Create: `.github/ISSUE_TEMPLATE/feature_request.md`
- Create: `.github/PULL_REQUEST_TEMPLATE.md`

- [ ] **Step 1: Create Governance Files**

```bash
mkdir -p .github/ISSUE_TEMPLATE
```

Create `CONTRIBUTING.md`:
```markdown
# Contributing to Unified ML Platform
We love your input! We want to make contributing to this project as easy and transparent as possible.
1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!
```

Create `SECURITY.md`:
```markdown
# Security Policy
## Supported Versions
| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
## Reporting a Vulnerability
Please report security vulnerabilities to hillaniljppatel@gmail.com.
```

Create `CODE_OF_CONDUCT.md`:
```markdown
# Contributor Covenant Code of Conduct
Please be respectful and considerate of others. Harassment will not be tolerated.
```

Create `CHANGELOG.md`:
```markdown
# Changelog
All notable changes to this project will be documented in this file.
## [Unreleased]
- Initial governance structure.
```

Create `CODEOWNERS`:
```text
* @STiFLeR7
```

Create `.github/ISSUE_TEMPLATE/bug_report.md`:
```markdown
---
name: Bug report
about: Create a report to help us improve
title: ''
labels: bug
assignees: ''
---
**Describe the bug**
A clear and concise description of what the bug is.
**To Reproduce**
Steps to reproduce the behavior.
**Expected behavior**
A clear description of what you expected to happen.
```

Create `.github/ISSUE_TEMPLATE/feature_request.md`:
```markdown
---
name: Feature request
about: Suggest an idea for this project
title: ''
labels: enhancement
assignees: ''
---
**Is your feature request related to a problem?**
**Describe the solution you'd like**
```

Create `.github/PULL_REQUEST_TEMPLATE.md`:
```markdown
## Description
Provide a brief summary of the changes.

## Related Issues
Fixes #

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Linting passed
```

- [ ] **Step 2: Commit**

```bash
git add CONTRIBUTING.md SECURITY.md CODE_OF_CONDUCT.md CHANGELOG.md CODEOWNERS .github/
git commit -m "docs: add repository governance and templates"
```

### Task 2: Configuration & DX

**Files:**
- Create: `backend/app/config.py`
- Create: `backend/.env.example`
- Modify: `backend/app/database.py:9-11`
- Create: `Makefile`
- Create: `.pre-commit-config.yaml`
- Create: `backend/pyproject.toml`
- Create: `.devcontainer/devcontainer.json`

- [ ] **Step 1: Create Centralized Configuration**

Create `backend/app/config.py`:
```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    app_name: str = "Unified ML Platform"
    version: str = "1.0.0"
    db_path: str = "data/platform.db"
    log_level: str = "INFO"
    
    model_config = SettingsConfigDict(env_file=".env")

@lru_cache()
def get_settings():
    return Settings()
```

Create `backend/.env.example`:
```env
APP_NAME="Unified ML Platform"
VERSION="1.0.0"
DB_PATH="data/platform.db"
LOG_LEVEL="INFO"
```

- [ ] **Step 2: Update Database to Use Config**

Modify `backend/app/database.py` lines 9-11:
```python
from .config import get_settings

def get_db_path():
    settings = get_settings()
    base_dir = os.path.dirname(os.path.dirname(__file__))
    if os.path.isabs(settings.db_path):
        return settings.db_path
    return os.path.join(base_dir, settings.db_path)

DB_PATH = get_db_path()
```

- [ ] **Step 3: Setup DX Tools**

Create `Makefile`:
```makefile
.PHONY: install test lint run-backend run-frontend

install:
	cd backend && pip install -r requirements.txt
	cd frontend && npm install

test:
	cd backend && pytest
	cd frontend && npm test

lint:
	cd backend && ruff check . && black --check .
```

Create `backend/pyproject.toml`:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100

[tool.black]
line-length = 100
```

Create `.pre-commit-config.yaml`:
```yaml
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
```

Create `.devcontainer/devcontainer.json`:
```json
{
    "name": "AutoML Dev",
    "image": "mcr.microsoft.com/devcontainers/python:3.11",
    "features": {
        "ghcr.io/devcontainers/features/node:1": {}
    },
    "postCreateCommand": "make install"
}
```

- [ ] **Step 4: Commit**

```bash
git add backend/app/config.py backend/.env.example backend/app/database.py Makefile backend/pyproject.toml .pre-commit-config.yaml .devcontainer/
git commit -m "chore: setup pydantic config and DX tools"
```

### Task 3: CI/CD & Release Engineering

**Files:**
- Create: `.github/workflows/backend.yml`
- Create: `.github/workflows/frontend.yml`
- Create: `.github/workflows/release.yml`

- [ ] **Step 1: GitHub Actions**

Create `.github/workflows/backend.yml`:
```yaml
name: Backend CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: cd backend && pip install -r requirements.txt pytest ruff black
      - run: cd backend && ruff check .
      - run: cd backend && black --check .
      - run: cd backend && pytest
```

Create `.github/workflows/frontend.yml`:
```yaml
name: Frontend CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '20'
      - run: cd frontend && npm ci
      - run: cd frontend && npm run build
```

Create `.github/workflows/release.yml`:
```yaml
name: Release
on:
  push:
    tags:
      - 'v*'
jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Create Release
        uses: softprops/action-gh-release@v1
        with:
          generate_release_notes: true
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/
git commit -m "ci: add github actions for backend, frontend and release"
```

### Task 4: Logging & Observability

**Files:**
- Create: `backend/app/logger.py`
- Modify: `backend/app/main.py:16-18`
- Modify: `backend/app/ml/trainer.py`

- [ ] **Step 1: Setup Structured Logger**

Create `backend/app/logger.py`:
```python
import logging
from .config import get_settings

def setup_logger(name: str):
    settings = get_settings()
    logger = logging.getLogger(name)
    logger.setLevel(settings.log_level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = setup_logger(__name__)
```

- [ ] **Step 2: Add Main API Logging**

Modify `backend/app/main.py` adding at the top:
```python
from .logger import setup_logger
logger = setup_logger(__name__)
```
Replace print statements in `lifespan`:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    init_db()
    logger.info("Database initialized")
    logger.info("Unified Supervised Learning Platform is ready!")
    yield
```

- [ ] **Step 3: Commit**

```bash
git add backend/app/logger.py backend/app/main.py
git commit -m "feat: implement structured logging"
```

### Task 5: Testing Setup

**Files:**
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/test_main.py`

- [ ] **Step 1: Backend Tests**

```bash
mkdir -p backend/tests
```
Create `backend/tests/__init__.py`:
```python
```
Create `backend/tests/test_main.py`:
```python
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"
```

- [ ] **Step 2: Run test to verify it passes**

```bash
cd backend && pip install pytest httpx pydantic-settings
pytest tests/test_main.py -v
```

- [ ] **Step 3: Commit**

```bash
git add backend/tests/
git commit -m "test: add initial backend tests"
```

### Task 6: Documentation

**Files:**
- Create: `docs/architecture.md`
- Create: `docs/troubleshooting.md`
- Create: `docs/deployment.md`
- Create: `docs/api.md`
- Create: `docs/adr/0001-initial-architecture.md`

- [ ] **Step 1: Create Docs**

```bash
mkdir -p docs/adr
```

Create `docs/architecture.md`:
```markdown
# Architecture
- **Backend:** FastAPI, scikit-learn
- **Frontend:** React, Vite, Tailwind CSS
- **Database:** SQLite
```

Create `docs/troubleshooting.md`:
```markdown
# Troubleshooting
- **Database locked:** Ensure no other process is holding the SQLite file.
- **Node errors:** Delete `node_modules` and run `npm ci`.
```

Create `docs/deployment.md`:
```markdown
# Deployment
1. Build frontend: `npm run build`
2. Start backend: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
```

Create `docs/api.md`:
```markdown
# API Reference
See `/docs` endpoint when running the app for Swagger UI.
```

Create `docs/adr/0001-initial-architecture.md`:
```markdown
# 1. Initial Architecture
**Date:** 2026-06-02
**Decision:** Use FastAPI for backend and React/Vite for frontend.
**Status:** Accepted
**Reason:** Rapid development, high performance, and great ecosystem.
```

- [ ] **Step 2: Commit**

```bash
git add docs/
git commit -m "docs: add comprehensive documentation"
```

### Task 7: ML Engineering Additions

**Files:**
- Modify: `backend/app/ml/trainer.py`
- Modify: `backend/requirements.txt`

- [ ] **Step 1: Add MLFlow Integration Hooks**

Update `backend/requirements.txt` to include `mlflow`:
```text
fastapi
uvicorn
scikit-learn
pandas
numpy
joblib
matplotlib
seaborn
python-multipart
pydantic
shap
fpdf2
scipy
xgboost
psutil
pydantic-settings
mlflow
```

- [ ] **Step 2: Commit**

```bash
git add backend/
git commit -m "feat: add ml engineering hooks and pydantic-settings"
```

---
