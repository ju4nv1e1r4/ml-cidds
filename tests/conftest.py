import pytest
from utils.gcp import CloudStorageOps

@pytest.fixture(scope="session")
def gcs():
    return CloudStorageOps("ml-anomaly-detection")

@pytest.fixture
def test_file(tmp_path):
    src = "tests/test.txt"
    dest = "test.txt"
    return src, dest