from unittest.mock import MagicMock, patch

import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.ml.metrics import SystemMetrics
from src.ml.optmize import Optimize
from src.models.load import Load
from utils.gcp import CloudStorageOps


@pytest.fixture(scope="session")
def gcs():
    return CloudStorageOps("ml-anomaly-detection")


@pytest.fixture(scope="session")
def load():
    return Load("preprocessed/CIDDS-001.csv")

@pytest.fixture(scope="session")
def optimize():
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
    }

    thresholds = {
        "recall": 0.90,
        "precision": 0.60,
        "roc_auc": 0.40,
        "f1_score": 0.75,
        "accuracy": 0.5,
    }

    X, y = make_classification(n_samples=100, n_features=5, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return Optimize(
        model=RandomForestClassifier(),
        param_grid=param_grid,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        cv=5,
        verbose=0,
        n_iter=5,
        scoring="recall",
        thresholds=thresholds,
    )

@pytest.fixture(scope="session")
@patch('pickle.load')
def monitoring_supervised_model(mock_pickle_load):
    model_path = "src/artifacts/model_v20250409_154518.pkl"
    mock_model = MagicMock()
    mock_model.predict.return_value = [0]
    mock_pickle_load.return_value = mock_model
    sup_sample_data = [
        1678886400.0, 
        1678886401.0, 
        2000.0, 
        100000000.0, 
        8000.0, 
        1.0
    ]

    infer_callable = mock_model.predict

    return SystemMetrics(
        infer_function=infer_callable,
        model_path=model_path,
        sample_data=sup_sample_data
    )

@pytest.fixture(scope="session")
@patch('pickle.load')
def monitoring_unsupervised_model(mock_pickle_load):
    model_path = "src/artifacts/unsupervised_model_v20250409_112404.pkl"
    mock_model = MagicMock()
    mock_model.predict.return_value = [0]
    mock_pickle_load.return_value = mock_model
    unsup_sample_data = [
        1678886400.0, 
        1678886401.0, 
        1, 
        200, 
        443.0, 
        "SYN", 
        0.002, 
        0.005, 
        "TCP"
    ]

    infer_callable = mock_model.predict

    return SystemMetrics(
        infer_function=infer_callable,
        model_path=model_path,
        sample_data=unsup_sample_data
    )

@pytest.fixture
def test_file(tmp_path):
    src = "tests/test.txt"
    dest = "test.txt"
    return src, dest
