import os

import pytest

os.environ["TESTING"] = "True"

from fastapi.testclient import TestClient

from app.database import get_db_path
from app.main import app

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


def test_db_path_is_test():
    test_db = get_db_path()
    assert "test_platform.db" in test_db
