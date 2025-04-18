import re
import pickle
from datetime import datetime
from utils.gcp import CloudStorageOps
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Model:
    @staticmethod
    def predict(model, features):
        try:
            prediction = model.predict(features)
            logging.info("Prediction made successfully")
            return prediction
        except Exception as e:
            logging.error(f"Error predicting: {e}")

    @staticmethod
    def build_features_supervised(json_data):
        start_session = json_data["start_session"]  # ISO: "2025-04-16T12:00:00"
        end_session = json_data["end_session"]  # ISO
        packets = json_data["packets"]  # int
        bytes_total = json_data["bytes"]  # int
        source_port = float(json_data["source_port"])  # float
        flag = json_data["flag"]  # example: "SYN,ACK"

        dt_start = datetime.fromisoformat(start_session)
        dt_end = datetime.fromisoformat(end_session)

        duration = (dt_end - dt_start).total_seconds()
        duration = max(duration, 0.001)

        total_packets_used = packets
        bytes_flow = bytes_total
        bytes_per_packet = bytes_total / packets if packets > 0 else 0
        packets_per_seconds = packets / duration if duration > 0 else 0

        hour_of_day = dt_start.hour

        common_ports = {80.0, 443.0, 22.0, 21.0}

        is_common_port = 1 if source_port in common_ports else 0

        flag_upper = flag.upper()
        has_SYN = 1 if "SYN" in flag_upper else 0
        has_ACK = 1 if "ACK" in flag_upper else 0
        has_RST = 1 if "RST" in flag_upper else 0
        has_FIN = 1 if "FIN" in flag_upper else 0

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
            has_FIN,
        ]

        return features

    @staticmethod
    def build_features_unsupervised(json_data, source_ip_freq=0, dest_ip_freq=0):
        start_session = json_data["start_session"]
        end_session = json_data["end_session"]
        packets = json_data["packets"]
        bytes_total = json_data["bytes"]
        source_port = float(json_data["source_port"])
        flag = json_data["flag"]
        protocol = json_data.get("protocol", "TCP")

        dt_start = datetime.fromisoformat(start_session)
        dt_end = datetime.fromisoformat(end_session)

        duration = (dt_end - dt_start).total_seconds()
        duration = max(duration, 0.001)

        total_packets_used = packets
        bytes_flow = bytes_total
        bytes_per_packet = bytes_total / packets if packets > 0 else 0
        packets_per_seconds = packets / duration if duration > 0 else 0

        hour_of_day = dt_start.hour

        common_ports = {80.0, 443.0, 22.0, 21.0}
        is_common_port = 1 if source_port in common_ports else 0

        flag_upper = flag.upper()
        has_SYN = 1 if "SYN" in flag_upper else 0
        has_ACK = 1 if "ACK" in flag_upper else 0
        has_RST = 1 if "RST" in flag_upper else 0
        has_FIN = 1 if "FIN" in flag_upper else 0

        is_hex_flag = 1 if flag.startswith("0x") else 0

        source_ip_freq = source_ip_freq
        dest_ip_freq = dest_ip_freq

        proto_upper = protocol.upper()
        network_protocol_ICMP = 1 if proto_upper == "ICMP" else 0
        network_protocol_TCP = 1 if proto_upper == "TCP" else 0
        network_protocol_UDP = 1 if proto_upper == "UDP" else 0

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
            has_FIN,
            is_hex_flag,
            source_ip_freq,
            dest_ip_freq,
            network_protocol_ICMP,
            network_protocol_TCP,
            network_protocol_UDP,
        ]

        return features

    @staticmethod
    def get_latest_model(prefix="src/artifacts/", supervised=True):
        try:
            gcs = CloudStorageOps("ml-anomaly-detection")
            all_files = gcs.list_from_bucket()

            if supervised == True:
                model_prefix = f"{prefix}model_v"
            elif supervised == False:
                model_prefix = f"{prefix}unsupervised_model_v"

            model_files = [
                f
                for f in all_files
                if f.startswith(model_prefix) and f.endswith(".pkl")
            ]
            logging.info(f"Model files found: {model_files[:5]}...")
            if not model_files:
                return None
        except Exception as load_model_error:
            logging.error(f"Error loading model: {load_model_error}")
            return None
        try:

            def extract_datetime(file_name):
                match = re.search(r"(\d{8}_\d{6})", file_name)
                if match:
                    return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
                return datetime.min

            latest_model = max(model_files, key=extract_datetime)
            logging.info("Latest model found")
            return latest_model
        except Exception as fin_latest_model_error:
            logging.error(f"Error finding latest model: {fin_latest_model_error}")
            return None

    @staticmethod
    def load_model(supervised=True):
        try:
            logging.info("Loading model...")
            gcs = CloudStorageOps("ml-anomaly-detection")
            model_path = Model.get_latest_model(supervised=supervised)
            model_data = gcs.load_from_bucket(model_path)

            if model_data is None:
                raise Exception(f"Model data not found at {model_path}")

            logging.info(f"Model loaded: {model_path}")
            model = pickle.loads(model_data)
            return model
        except Exception as load_model_error:
            logging.error(f"Error loading model: {load_model_error}")
            raise load_model_error
