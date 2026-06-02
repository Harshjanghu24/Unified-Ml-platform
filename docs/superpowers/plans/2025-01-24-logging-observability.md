# Logging & Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement structured logging across the backend to improve traceability and debugging capabilities.

**Architecture:** Create a centralized logging utility in `backend/app/logger.py` that configures a standard Python logger based on application settings. Integrate this logger into the main FastAPI application lifespan and the ML training engine to provide high-signal visibility into system state and training progress.

**Tech Stack:** Python `logging` module, FastAPI, Pydantic Settings.

---

### Task 1: Setup Centralized Logger

**Files:**
- Create: `backend/app/logger.py`

- [ ] **Step 1: Create the logger module**
Create `backend/app/logger.py` with the `setup_logger` function and a default instance.

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

- [ ] **Step 2: Verify logger creation**
Run a simple script to verify the logger can be imported and respects the log level.
Run: `python -c "from backend.app.logger import logger; logger.info('Logger initialized')"`
Expected: Output containing `INFO - backend.app.logger - Logger initialized`

### Task 2: Integrate Logger into FastAPI Main

**Files:**
- Modify: `backend/app/main.py`

- [ ] **Step 1: Import logger and replace print statements**
Replace the `print` statements in the `lifespan` function with `logger.info`.

```python
from .logger import setup_logger
logger = setup_logger(__name__)

# ... in lifespan ...
    logger.info("Database initialized")
    logger.info("Unified Supervised Learning Platform is ready!")
```

- [ ] **Step 2: Verify API logging**
Start the backend server and check the output.
Run: `cd backend && uvicorn app.main:app --reload`
Expected: Log messages showing "Database initialized" and "Unified Supervised Learning Platform is ready!" in the structured format.

### Task 3: Add Training Lifecycle Logs

**Files:**
- Modify: `backend/app/ml/trainer.py`

- [ ] **Step 1: Import logger and add start/end logs**
Add logging at the beginning and end of the `train_models` function.

```python
from ..logger import setup_logger
logger = setup_logger(__name__)

# ... in train_models ...
    logger.info(f"Starting model training for problem type: {problem_type}")
    # ... at the end ...
    logger.info(f"Completed training for {len(results)} models. Best model: {results[best_idx]['model_name']}")
```

- [ ] **Step 2: Add per-model progress logs**
Log when starting training for each specific model.

```python
# ... inside the loop in train_models ...
    for model_name, model_class in model_configs:
        logger.info(f"Training model: {model_name}")
```

- [ ] **Step 3: Verify training logs**
Run a training task (or a test that triggers training) and verify the logs appear.
Run: `pytest backend/tests/test_trainer.py` (assuming tests exist, otherwise manual verification via API)
Expected: Logs showing "Starting model training...", "Training model: ...", and "Completed training...".

### Task 4: Final Verification and Commit

- [ ] **Step 1: Run all backend tests**
Run: `cd backend && pytest`
Expected: All tests pass.

- [ ] **Step 2: Commit changes**
Run: `git add backend/app/logger.py backend/app/main.py backend/app/ml/trainer.py`
Run: `git commit -m "feat: implement structured logging"`
