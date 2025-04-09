from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    recall_score,
    precision_score,
    f1_score
)
from sklearn.model_selection import train_test_split
from google.resumable_media.common import InvalidResponse
from colorama import Fore, init
import pandas as pd
import io
import logging

from utils.gcp import CloudStorageOps

init(autoreset=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


PATH_FILE = "preprocessed/CIDDS-001.csv"
gcs = CloudStorageOps("ml-anomaly-detection")

try:
    load = gcs.load_from_bucket(PATH_FILE)
    logging.info(Fore.GREEN + f"File loaded successfully: {PATH_FILE}.")
    print(" ")
except InvalidResponse as load_error:
    logging.error(Fore.RED + f"Error: {load_error}")
    print(" ")

try:
    if load is not None:
        df = pd.read_csv(io.BytesIO(load))
    else:
        logging.error("Load returned None, cannot read CSV.")
        print(" ")
    logging.info(Fore.BLUE + f"File loaded: {PATH_FILE}.")
    print(" ")
except FileNotFoundError as file_error:
    logging.error(Fore.RED + f"Error: {file_error}")
    print(" ")

try:
    X = df.drop(columns=["class", "is_attack"])
    y = df["is_attack"]
    logging.info(Fore.BLUE + "Features and Target splitted.")
    print(" ")
except Exception as cols_error:
    logging.error(Fore.RED + f"Error: {cols_error}")
    print(" ")

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )
    logging.info(Fore.BLUE + f"Data successfully splitted -> Train: {X_train.shape} || Test: {X_test.shape}")
    print(" ")
except Exception as split_error:
    logging.error(Fore.RED + f"Error: {split_error}")
    print(" ")


print("---\n")
print(Fore.YELLOW + "Starting experiments with IsolationForest...")
print(" ")

model = IsolationForest(
    n_estimators=100,
    max_samples='auto',
    contamination=0.5,
    max_features=1.0,
    random_state=42
)

model.fit(X_train)

y_pred_raw = model.predict(X_test)
y_pred = [1 if val == -1 else 0 for val in y_pred_raw]

print("\nClassification Report\n", classification_report(y_test, y_pred))
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred):.4f}")

print(Fore.GREEN + "Experiment is done")