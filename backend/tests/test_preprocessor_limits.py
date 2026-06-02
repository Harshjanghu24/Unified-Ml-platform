import numpy as np
import pandas as pd

from app.ml.preprocessor import build_preprocessing_pipeline


def test_high_cardinality_drop():
    # Create data with high cardinality categorical column (> 20 unique values)
    # Plus a stable numeric feature so it doesn't crash
    data = {
        "high_card": [f"val_{i}" for i in range(100)],
        "stable_feat": np.random.rand(100),
        "target": [0, 1] * 50,
    }
    df = pd.DataFrame(data)

    result = build_preprocessing_pipeline(df, "target")

    # Verify high_card WAS dropped.
    assert "high_card" not in result["X_raw"].columns
    assert "stable_feat" in result["feature_names"]


def test_id_column_drop():
    # Create data with an ID-like column
    data = {"user_id": [i for i in range(100)], "feat1": np.random.rand(100), "target": [0, 1] * 50}
    df = pd.DataFrame(data)

    result = build_preprocessing_pipeline(df, "target")

    # Verify user_id WAS dropped.
    assert "user_id" not in result["X_raw"].columns
    assert "feat1" in result["feature_names"]


def test_feature_truncation():
    # Create data that will expand to > 500 features
    # 1000 rows to keep cardinality ratio low (10/1000 = 0.01 < 0.05)
    data = {"stable_feat": np.random.rand(1000)}
    for i in range(60):
        data[f"cat_{i}"] = [f"val_{j}" for j in range(10)] * 100
    data["target"] = [0, 1] * 500
    df = pd.DataFrame(data)

    result = build_preprocessing_pipeline(df, "target", "binary")

    # Verify truncated to 500
    assert len(result["feature_names"]) == 500
