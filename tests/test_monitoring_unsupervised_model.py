import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

def test_model_size(monitoring_unsupervised_model):
    size_mb = monitoring_unsupervised_model.model_size()

    assert isinstance(size_mb, float)
    assert size_mb > 0.0
    assert size_mb <= 65.0

def test_latency(monitoring_unsupervised_model):
    latency = monitoring_unsupervised_model.latency()

    assert isinstance(latency, float)
    assert latency > 0.0
    assert latency < 0.10

def test_throughput(monitoring_unsupervised_model):
    throughput = monitoring_unsupervised_model.throughput()

    assert isinstance(throughput, float)
    assert isinstance > 0.0
    assert isinstance < 100.0

def test_export_metrics(monitoring_unsupervised_model):
    local_path, filename = monitoring_unsupervised_model.export_metrics(mode="supervised")

    assert isinstance(local_path, str)
    assert isinstance(filename, str)
    assert os.path.exists(local_path)
    assert filename.startswith("metrics_unsupervised_")
    assert filename.endswith(".json")