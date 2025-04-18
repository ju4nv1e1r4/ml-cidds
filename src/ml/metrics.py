import os
import time
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class SystemMetrics:
    def __init__(self):
        pass

    def model_size(self):
        model_path = "model-here.pkl"  # *** ALTER MODEL HERE ***
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        logging.info(f"Model size: {size_mb:.2f}MB")
        return size_mb

    def latency(self):
        data_to_infer = ["data", "to", "infer"] * 100
        start_time = time.time()
        # INFERENCE HERE
        end_time = time.time()

        latency = (end_time - start_time) / len(data_to_infer)
        logging.info(f"Latency: {latency:.6f} seconds per inference")
        return latency

    def throughput(self):
        data_to_infer = ["data", "to", "infer"] * 1000
        start_time = time.time()
        # INFERENCE HERE
        end_time = time.time()

        total_time = end_time - start_time
        throughput = len(data_to_infer) / total_time

        logging.info(f"Throughput: {throughput:.2f} inferences per second")
        return throughput
