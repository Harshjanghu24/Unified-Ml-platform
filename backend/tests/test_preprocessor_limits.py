import numpy as np
import pandas as pd

from app.ml.preprocessor import build_preprocessing_pipeline


def test_no_high_cardinality_drop():
    # Create data with high cardinality categorical column (> 20 unique values)
    # 100 unique values
    data = {"high_card": [f"val_{i}" for i in range(100)], "target": [0, 1] * 50}
    df = pd.DataFrame(data)

    result = build_preprocessing_pipeline(df, "target")

    # Verify high_card was NOT dropped.
    # Since it is categorical, it should be one-hot encoded in X_raw.
    assert any(col.startswith("high_card_") for col in result["X_raw"].columns)


def test_no_id_column_drop():
    # Create data with an ID-like column
    # 100% unique, named with 'id'
    data = {"user_id": [i for i in range(100)], "feat1": np.random.rand(100), "target": [0, 1] * 50}
    df = pd.DataFrame(data)

    result = build_preprocessing_pipeline(df, "target")

    # Current behavior: user_id is dropped
    # Desired behavior: user_id is kept
    assert "user_id" in result["X_raw"].columns


def test_no_feature_truncation():
    # Create data that will expand to > 500 features
    # 60 categorical columns with 10 categories each -> 600 features
    # Length 100
    data = {}
    for i in range(60):
        data[f"cat_{i}"] = [f"val_{j}" for j in range(10)] * 10
    data["target"] = [0, 1] * 50
    df = pd.DataFrame(data)

    result = build_preprocessing_pipeline(df, "target")

    # Current behavior: truncated to 500
    # Desired behavior: not truncated (should be around 600)
    assert len(result["feature_names"]) > 500
