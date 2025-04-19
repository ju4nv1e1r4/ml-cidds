import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def test_load_file_from_bucket(gcs):
    file_path = "src/artifacts/model_v20250409_092346_metadata.json"
    content = gcs.load_from_bucket(file_path)
    assert isinstance(content, bytes)


def test_list_files_on_bucket(gcs):
    files = gcs.list_from_bucket()
    assert files is not None
    assert isinstance(files, list)


def test_load_file_to_bucket(gcs, test_file):
    src, dest = test_file
    gcs.upload_to_bucket(src, dest)
    files = gcs.list_from_bucket()
    assert dest in files


def test_delete_file_from_bucket(gcs, test_file):
    src, dest = test_file
    gcs.upload_to_bucket(src, dest)
    gcs.delete_from_bucket(dest)
    files = gcs.list_from_bucket()
    assert dest not in files
