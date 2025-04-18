import pytest
from utils.gcp import CloudStorageOps
from src.models.load import Load


@pytest.fixture(scope="session")
def gcs():
    return CloudStorageOps("ml-anomaly-detection")


@pytest.fixture(scope="session")
def load():
    return Load("preprocessed/CIDDS-001.csv")


@pytest.fixture
def test_file(tmp_path):
    src = "tests/test.txt"
    dest = "test.txt"
    return src, dest
