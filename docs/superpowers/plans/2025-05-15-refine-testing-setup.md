# Test Setup Refinement Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enhance test isolation by using a dedicated test database for the backend and Mock Service Worker (MSW) for the frontend.

**Architecture:** 
- Backend: Use environment variables to switch to a temporary SQLite database during testing.
- Frontend: Use MSW to mock API responses in Vitest, ensuring tests don't depend on a running backend.

**Tech Stack:** FastAPI, pytest, MSW, Vitest.

---

### Task 1: Backend Test Database Isolation

**Files:**
- Modify: `backend/app/config.py`
- Modify: `backend/app/database.py`
- Modify: `backend/tests/test_main.py`

- [ ] **Step 1: Update Settings to include testing flag**

Modify `backend/app/config.py`:
```python
class Settings(BaseSettings):
    app_name: str = "Unified ML Platform"
    version: str = "1.0.0"
    db_path: str = "data/platform.db"
    log_level: str = "INFO"
    testing: bool = False  # Add this field
    
    model_config = SettingsConfigDict(env_file=".env")
```

- [ ] **Step 2: Make database path dynamic**

Modify `backend/app/database.py`:
Remove `DB_PATH` constant and update `get_db_path` and `get_connection`.

```python
def get_db_path():
    settings = get_settings()
    path = settings.db_path
    if settings.testing:
        path = "data/test_platform.db"
    
    base_dir = os.path.dirname(os.path.dirname(__file__))
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)

# Remove DB_PATH = get_db_path()

def get_connection():
    """Get a SQLite connection with row_factory enabled."""
    db_path = get_db_path()
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn
```

- [ ] **Step 3: Update `backend/tests/test_main.py` to use test environment**

```python
import os
import pytest
os.environ["TESTING"] = "True"

from fastapi.testclient import TestClient
from app.main import app
from app.database import get_db_path

client = TestClient(app)

@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    # Setup: logic to ensure we are using test DB
    yield
    # Cleanup: remove test DB file after tests
    test_db = get_db_path()
    if os.path.exists(test_db):
        os.remove(test_db)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"
```

- [ ] **Step 4: Run backend tests**

Run: `pytest backend/tests/test_main.py`
Expected: PASS and `data/test_platform.db` should be created and then deleted.

- [ ] **Step 5: Commit backend changes**

```bash
git add backend/app/config.py backend/app/database.py backend/tests/test_main.py
git commit -m "test: isolate backend tests with temporary SQLite database"
```

---

### Task 2: Frontend MSW Mocking Setup

**Files:**
- Modify: `frontend/package.json`
- Create: `frontend/src/mocks/handlers.js`
- Create: `frontend/src/mocks/server.js`
- Modify: `frontend/src/setupTests.js`

- [ ] **Step 1: Install MSW**

Run: `npm install msw --save-dev` in `frontend` directory.

- [ ] **Step 2: Create MSW Handlers**

Create `frontend/src/mocks/handlers.js`:
```javascript
import { http, HttpResponse } from 'msw'

export const handlers = [
  http.get('http://localhost:8000/', () => {
    return HttpResponse.json({
      name: "Unified Supervised Learning Platform",
      version: "1.0.0",
      status: "running",
      docs: "/docs",
    })
  }),
]
```

- [ ] **Step 3: Create MSW Server**

Create `frontend/src/mocks/server.js`:
```javascript
import { setupServer } from 'msw/node'
import { handlers } from './handlers'

export const server = setupServer(...handlers)
```

- [ ] **Step 4: Configure setupTests.js**

Modify `frontend/src/setupTests.js`:
```javascript
import '@testing-library/jest-dom'
import { beforeAll, afterEach, afterAll } from 'vitest'
import { server } from './mocks/server'

// Establish API mocking before all tests.
beforeAll(() => server.listen())

// Reset any request handlers that we may add during the tests,
// so they don't affect other tests.
afterEach(() => server.resetHandlers())

// Clean up after the tests are finished.
afterAll(() => server.close())
```

- [ ] **Step 5: Run frontend tests**

Run: `npm test` in `frontend` directory.
Expected: PASS.

- [ ] **Step 6: Commit frontend changes**

```bash
git add frontend/package.json frontend/src/mocks/handlers.js frontend/src/mocks/server.js frontend/src/setupTests.js
git commit -m "test: add MSW mocking for frontend tests"
```
