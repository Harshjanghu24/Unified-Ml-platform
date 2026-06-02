import pytest
import pandas as pd
import numpy as np
import sys
from unittest.mock import MagicMock, patch

# Mock mlflow before importing trainer if it's not installed
mock_mlflow = MagicMock()
mock_mlflow_sklearn = MagicMock()
if 'mlflow' not in sys.modules:
    sys.modules['mlflow'] = mock_mlflow
    sys.modules['mlflow.sklearn'] = mock_mlflow_sklearn

from app.ml.trainer import train_models

def test_train_models_returns_reproducibility_metadata():
    # Setup simple binary classification data
    X_train = pd.DataFrame(np.random.rand(10, 2), columns=['f1', 'f2'])
    X_test = pd.DataFrame(np.random.rand(5, 2), columns=['f1', 'f2'])
    y_train = pd.Series([0, 1] * 5)
    y_test = pd.Series([0, 1, 0, 1, 0])
    problem_type = "binary"

    results = train_models(X_train, X_test, y_train, y_test, problem_type)

    assert len(results) > 0
    for res in results:
        assert "reproducibility_metadata" in res
        metadata = res["reproducibility_metadata"]
        assert "python_version" in metadata
        assert "sklearn_version" in metadata
        assert "pandas_version" in metadata
        assert "numpy_version" in metadata

@patch("app.ml.trainer.HAS_MLFLOW", True)
def test_train_models_mlflow_integration():
    # Setup simple binary classification data
    X_train = pd.DataFrame(np.random.rand(10, 2), columns=['f1', 'f2'])
    X_test = pd.DataFrame(np.random.rand(5, 2), columns=['f1', 'f2'])
    y_train = pd.Series([0, 1] * 5)
    y_test = pd.Series([0, 1, 0, 1, 0])
    problem_type = "binary"

    with patch("app.ml.trainer.mlflow") as mock_mlflow_mod:
        # Configure mocks
        mock_mlflow_mod.active_run.return_value = None
        
        train_models(X_train, X_test, y_train, y_test, problem_type)

        # Verify mlflow was called
        assert mock_mlflow_mod.start_run.called
        assert mock_mlflow_mod.log_params.called
        assert mock_mlflow_mod.log_metrics.called
        assert mock_mlflow_mod.sklearn.log_model.called

@patch("app.ml.trainer.HAS_MLFLOW", False)
def test_train_models_no_mlflow_graceful():
    # Ensure it doesn't crash if MLflow is not present
    X_train = pd.DataFrame(np.random.rand(10, 2), columns=['f1', 'f2'])
    X_test = pd.DataFrame(np.random.rand(5, 2), columns=['f1', 'f2'])
    y_train = pd.Series([0, 1] * 5)
    y_test = pd.Series([0, 1, 0, 1, 0])
    problem_type = "binary"

    # This should not raise any exceptions
    results = train_models(X_train, X_test, y_train, y_test, problem_type)
    assert len(results) > 0
