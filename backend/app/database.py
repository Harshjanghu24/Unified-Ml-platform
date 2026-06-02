"""
Database module for SQLite operations.
Handles connection management and schema initialization for
datasets and trained_models tables.
"""

import json
import os
import sqlite3
from datetime import datetime

from .config import get_settings


def get_db_path():
    settings = get_settings()
    path = settings.db_path
    if settings.testing:
        path = "data/test_platform.db"

    base_dir = os.path.dirname(os.path.dirname(__file__))
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)


def get_connection():
    """Get a SQLite connection with row_factory enabled."""
    db_path = get_db_path()
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database schema."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            target_column TEXT,
            problem_type TEXT,
            num_rows INTEGER,
            num_cols INTEGER,
            upload_time TEXT,
            summary TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trained_models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER,
            model_name TEXT NOT NULL,
            problem_type TEXT,
            metrics TEXT,
            training_time REAL,
            model_path TEXT,
            is_best INTEGER DEFAULT 0,
            best_params TEXT,
            cv_scores TEXT,
            created_at TEXT,
            FOREIGN KEY (dataset_id) REFERENCES datasets(id)
        )
    """)

    conn.commit()
    conn.close()


# ── CRUD helpers ──────────────────────────────────────────────


def save_dataset_record(filename, target_column, problem_type, num_rows, num_cols, summary):
    """Insert a dataset record and return its id."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO datasets (
            filename, target_column, problem_type, num_rows, num_cols, upload_time, summary
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            filename,
            target_column,
            problem_type,
            num_rows,
            num_cols,
            datetime.now().isoformat(),
            json.dumps(summary),
        ),
    )
    conn.commit()
    dataset_id = cursor.lastrowid
    conn.close()
    return dataset_id


def save_model_record(
    dataset_id,
    model_name,
    problem_type,
    metrics,
    training_time,
    model_path,
    is_best=False,
    best_params=None,
    cv_scores=None,
):
    """Insert a trained model record and return its id."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO trained_models
           (dataset_id, model_name, problem_type, metrics, training_time,
            model_path, is_best, best_params, cv_scores, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            dataset_id,
            model_name,
            problem_type,
            json.dumps(metrics),
            training_time,
            model_path,
            int(is_best),
            json.dumps(best_params) if best_params else None,
            json.dumps(cv_scores) if cv_scores else None,
            datetime.now().isoformat(),
        ),
    )
    conn.commit()
    model_id = cursor.lastrowid
    conn.close()
    return model_id


def get_all_models():
    """Return all saved model records."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM trained_models ORDER BY created_at DESC")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    for r in rows:
        r["metrics"] = json.loads(r["metrics"]) if r["metrics"] else {}
        r["best_params"] = json.loads(r["best_params"]) if r["best_params"] else {}
        r["cv_scores"] = json.loads(r["cv_scores"]) if r["cv_scores"] else {}
    return rows


def delete_model_record(model_id):
    """Delete a model record by id. Returns the model_path for file cleanup."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT model_path FROM trained_models WHERE id = ?", (model_id,))
    row = cursor.fetchone()
    if row is None:
        conn.close()
        return None
    model_path = row["model_path"]
    cursor.execute("DELETE FROM trained_models WHERE id = ?", (model_id,))
    conn.commit()
    conn.close()
    return model_path


def get_latest_dataset():
    """Return the most recently uploaded dataset record."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM datasets ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    if row:
        d = dict(row)
        d["summary"] = json.loads(d["summary"]) if d["summary"] else {}
        return d
    return None


def get_models_by_dataset(dataset_id):
    """Return all models trained on a specific dataset."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM trained_models WHERE dataset_id = ? ORDER BY created_at DESC", (dataset_id,)
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    for r in rows:
        r["metrics"] = json.loads(r["metrics"]) if r["metrics"] else {}
        r["best_params"] = json.loads(r["best_params"]) if r["best_params"] else {}
        r["cv_scores"] = json.loads(r["cv_scores"]) if r["cv_scores"] else {}
    return rows
