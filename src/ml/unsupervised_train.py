from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    recall_score,
    precision_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from google.resumable_media.common import InvalidResponse
from colorama import Fore, init
import pandas as pd
import io
import os
import json
import pickle
import datetime
import logging

from utils.gcp import CloudStorageOps

init(autoreset=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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
    X = df.drop(columns=["Unnamed: 0", "class", "is_attack"])
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
    logging.info(
        Fore.BLUE
        + f"Data successfully splitted -> Train: {X_train.shape} || Test: {X_test.shape}"
    )
    print(" ")
except Exception as split_error:
    logging.error(Fore.RED + f"Error: {split_error}")
    print(" ")


print("---\n")
print(Fore.YELLOW + "Starting experiments with IsolationForest...")
print(" ")

model = IsolationForest(
    n_estimators=100,
    max_samples="auto",
    contamination=0.5,
    max_features=1.0,
    random_state=42,
)

model.fit(X_train)

y_pred_raw = model.predict(X_test)
y_pred = [1 if val == -1 else 0 for val in y_pred_raw]

print("\nClassification Report\n", classification_report(y_test, y_pred))
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred):.4f}")

print(Fore.GREEN + "Experiment is done")

MODEL_BASE_PATH = "src/artifacts/"
MODEL_FILENAME = "unsupervised_model.pkl"
MODEL_VERSION_PREFIX = "unsupervised_model_v"

metadata = {
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred),
    "features": list(X.columns),
    "training_size": len(X_train),
    "test_size": len(X_test),
    "timestamp": datetime.datetime.now().isoformat(),
}

os.makedirs(MODEL_BASE_PATH, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_version = f"{MODEL_VERSION_PREFIX}{timestamp}"

model_path = os.path.join(MODEL_BASE_PATH, f"{model_version}.pkl")

try:
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    current_model_path = os.path.join(MODEL_BASE_PATH, MODEL_FILENAME)
    with open(current_model_path, "wb") as f:
        pickle.dump(model, f)

    metadata_path = os.path.join(MODEL_BASE_PATH, f"{model_version}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    current_version_path = os.path.join(MODEL_BASE_PATH, "current_version.txt")
    with open(current_version_path, "w") as f:
        f.write(model_version)

    logging.info(Fore.GREEN + f"Model saved successfully: {model_path}")
    logging.info(Fore.GREEN + f"Metatada saved: {metadata_path}")
    logging.info(Fore.GREEN + f"Model set as current: {current_model_path}")

    model_path_on_bucket = model_path
    metadata_path_on_bucket = metadata_path
    gcs.upload_to_bucket(current_model_path, model_path_on_bucket)
    gcs.upload_to_bucket(metadata_path, metadata_path_on_bucket)
    logging.info(Fore.BLUE + f"Model saved on bucket: {model_path_on_bucket}")
    logging.info(Fore.BLUE + f"Metadata saved on bucket: {metadata_path_on_bucket}")
except Exception as e:
    logging.exception(f"Error saving model: {e}")
