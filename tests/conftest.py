import pytest

from src.models.load import Load
from utils.gcp import CloudStorageOps


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
