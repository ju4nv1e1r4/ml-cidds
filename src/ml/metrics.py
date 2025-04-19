import json
import logging
import os
import time
import warnings
from datetime import datetime

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class SystemMetrics:
    def __init__(self, infer_function, model_path, sample_data):
        self.infer = infer_function
        self.model_path = model_path
        self.sample_data = sample_data

    def model_size(self):
        size_mb = os.path.getsize(self.model_path) / (1024 * 1024)
        logging.info(f"Model size: {size_mb:.2f}MB")
        return size_mb

    def latency(self):
        data = self.sample_data * 100
        start_time = time.time()
        for item in data:
            self.infer([item])
        end_time = time.time()
        latency = (end_time - start_time) / len(data)
        logging.info(f"Latency: {latency:.6f} seconds per inference")
        return latency

    def throughput(self):
        data = self.sample_data * 100
        start_time = time.time()
        for item in data:
            self.infer([item])
        end_time = time.time()
        total_time = end_time - start_time
        throughput = len(data) / total_time
        logging.info(f"Throughput: {throughput:.2f} inferences per second")
        return throughput

    def export_metrics(self, mode="supervised"):
        metrics = {
            "model_size_mb": self.model_size(),
            "latency_s": self.latency(),
            "throughput_ips": self.throughput(),
            "timestamp": datetime.utcnow().isoformat(),
            "mode": mode,
        }

        filename = f"metrics_{mode}_{int(time.time())}.json"
        local_path = f"data/metrics/{filename}"
        os.makedirs("metrics", exist_ok=True)

        with open(local_path, "w") as f:
            json.dump(metrics, f, indent=4)

        logging.info(f"Metrics exported: {local_path}")
        return local_path, filename
