from google.resumable_media.common import InvalidResponse
from colorama import init, Fore
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pandas as pd
import logging
import io

from utils.gcp import CloudStorageOps

init(autoreset=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
gcs = CloudStorageOps("ml-anomaly-detection")

class Load:
    def __init__(self, file_path):
        self.FILE_PATH = file_path

    def load_dataset(self) -> pd.DataFrame:
        try:
            load = gcs.load_from_bucket(self.FILE_PATH)
            logging.info(Fore.GREEN + f"File loaded successfully: {self.FILE_PATH}.")
            print("Tipo de load:", type(load))
            print("Conteúdo bruto:", load[:100])
            print("---")
        except InvalidResponse as load_error:
            logging.error(Fore.RED + f"Error: {load_error}")

        try:
            if load is not None:
                df = pd.read_csv(io.BytesIO(load))
                return df
            else:
                logging.error("Load returned None, cannot read CSV.")
            logging.info(Fore.BLUE + f"File loaded: {self.FILE_PATH}.")
        except FileNotFoundError as file_error:
            logging.error(Fore.RED + f"Error: {file_error}")

    def split_dataset(self, df: pd.DataFrame):
        try:
            safe_features = [
                "duration", "total_packets_used", "bytes_flow", "bytes_per_packet",
                "packets_per_seconds", "hour_of_day", "is_common_port",
                "has_SYN", "has_ACK", "has_RST", "has_FIN"
            ]
            X = df[safe_features]
            y = df["is_attack"]
            logging.info(Fore.BLUE + "Features and Target splitted.")
        except Exception as cols_error:
            logging.error(Fore.RED + f"Error: {cols_error}")

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.3,
                random_state=42,
                stratify=y,
                shuffle=True
            )
            y_train_shuffled = shuffle(y_train, random_state=42)
            logging.info(Fore.BLUE + f"Data successfully splitted -> Train: {X_train.shape} || Test: {X_test.shape}")
            return X_train, X_test, y_train_shuffled, y_test
        except Exception as split_error:
            logging.error(Fore.RED + f"Error: {split_error}")