import re
import pickle
from datetime import datetime
from utils.gcp import CloudStorageOps

class Model:
    @staticmethod
    def predict(model, features):
        return model.predict(features)
    
    @staticmethod
    def process_input_data(json_data):
        start_session = json_data['start_session']      # ISO: "2025-04-16T12:00:00"
        end_session = json_data['end_session']          # ISO
        packets = json_data['packets']                  # int
        bytes_total = json_data['bytes']                # int
        source_port = float(json_data['source_port'])   # float
        flag = json_data['flag']                        # example: "SYN,ACK"
        
        dt_start = datetime.fromisoformat(start_session)
        dt_end = datetime.fromisoformat(end_session)
        
        duration = (dt_end - dt_start).total_seconds()
        duration = max(duration, 0.001)

        total_packets_used = packets
        bytes_flow = bytes_total
        bytes_per_packet = bytes_total / packets if packets > 0 else 0
        packets_per_seconds = packets / duration if duration > 0 else 0

        hour_of_day = dt_start.hour

        common_ports = {
            80.0,
            443.0,
            22.0,
            21.0
        }

        is_common_port = 1 if source_port in common_ports else 0

        flag_upper = flag.upper()
        has_SYN = 1 if 'SYN' in flag_upper else 0
        has_ACK = 1 if 'ACK' in flag_upper else 0
        has_RST = 1 if 'RST' in flag_upper else 0
        has_FIN = 1 if 'FIN' in flag_upper else 0

        features = [
            duration,
            total_packets_used,
            bytes_flow,
            bytes_per_packet,
            packets_per_seconds,
            hour_of_day,
            is_common_port,
            has_SYN,
            has_ACK,
            has_RST,
            has_FIN
        ]

        return features
    
    @staticmethod
    def get_latest_model(prefix="src/artifacts", supervised=True):
        gcs = CloudStorageOps("ml-anomaly-detection")
        all_files = gcs.list_from_bucket()

        if supervised:
            model_prefix = f"{prefix}model_v"
        else:
            model_prefix = f"{prefix}unsupervised_model_v"

        model_files = [
            f for f in all_files
            if f.startswith(model_prefix) and f.endswith(".pkl")
        ]

        if not model_files:
            return None
        
        def extract_datetime(file_name):
            match = re.search(r"(\d{8}_\d{6})", file_name)
            if match:
                return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
            return datetime.min
        
        latest_model = max(model_files, key=extract_datetime)
        return latest_model
    
    @staticmethod
    def load_model(model_path):
        gcs = CloudStorageOps("ml-anomaly-detection")
        model_data = gcs.load_from_bucket(model_path)
        
        if model_data is None:
            raise Exception(f"Model data not found at {model_path}")
        
        model = pickle.loads(model_data)
        return model