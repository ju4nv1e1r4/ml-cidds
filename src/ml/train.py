import datetime
import io
import json
import logging
import os
import pickle
import time

import pandas as pd
from colorama import Fore, init
from google.resumable_media.common import InvalidResponse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from utils.constants import Config
from utils.gcp import CloudStorageOps

init(autoreset=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
c = Config()
logger = c.logger

PATH_FILE = "preprocessed/CIDDS-001.csv"
gcs = CloudStorageOps("ml-anomaly-detection")

try:
    load = gcs.load_from_bucket(PATH_FILE)
    logging.info(Fore.GREEN + f"File loaded successfully: {PATH_FILE}.")
except InvalidResponse as load_error:
    logging.error(Fore.RED + f"Error: {load_error}")

try:
    if load is not None:
        df = pd.read_csv(io.BytesIO(load))
    else:
        logging.error("Load returned None, cannot read CSV.")
    logging.info(Fore.BLUE + f"File loaded: {PATH_FILE}.")
except FileNotFoundError as file_error:
    logging.error(Fore.RED + f"Error: {file_error}")

try:
    safe_features = [
        "duration",
        "total_packets_used",
        "bytes_flow",
        "bytes_per_packet",
        "packets_per_seconds",
        "hour_of_day",
        "is_common_port",
        "has_SYN",
        "has_ACK",
        "has_RST",
        "has_FIN",
    ]
    X = df[safe_features]
    y = df["is_attack"]
    logging.info(Fore.BLUE + "Features and Target splitted.")
except Exception as cols_error:
    logging.error(Fore.RED + f"Error: {cols_error}")

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y, shuffle=True
    )
    y_train_shuffled = shuffle(y_train, random_state=42)
    logging.info(
        Fore.BLUE
        + f"Data successfully splitted -> Train: {X_train.shape}; Test: {X_test.shape}"
    )
except Exception as split_error:
    logging.error(Fore.RED + f"Error: {split_error}")

print("---\n")
logging.info(Fore.YELLOW + "Starting training with RandomForestClassifier...")

try:
    model = RandomForestClassifier(
        n_estimators=210,
        min_samples_split=10,
        min_samples_leaf=3,
        max_depth=20,
        max_features="sqrt",
        criterion="gini",
        bootstrap=False,
    )
    model.fit(X_train, y_train_shuffled)
    logging.info(Fore.GREEN + "Model training is complete.")
except Exception as fit_error:
    logging.error(Fore.RED + f"Error: {fit_error}")

try:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    rocauc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    logging.info(Fore.GREEN + "Predictions is done. Check this metrics:\n")
except Exception as pred_and_eval_error:
    logging.error(Fore.RED + f"Error: {pred_and_eval_error}")

print("1. Classification Report\n", cr)
print("\n2. Confusion Matrix\n", cm)
print("\n3. ROC AUC Score: ", rocauc)
print("\n4. Accuracy Score: ", acc)
print("\n5. Precision Score: ", precision)
print("\n6. Recall Score: ", recall)
print("\n7. F1 Score: ", f1)

MODEL_METRICS_PATH = f"data/logs/{int(time.time() * 1e6)}.json"

model_metrics = {
    "accuracy": acc,
    "precision": precision,
    "recall": recall,
    "fi_score": f1,
    "roc_auc_score": rocauc,
}

logger.info(MODEL_METRICS_PATH, **model_metrics)
os.makedirs(os.path.dirname(MODEL_METRICS_PATH), exist_ok=True)
json_log = json.dumps(model_metrics, indent=2)
open(MODEL_METRICS_PATH, "w").write(json_log)

MODEL_BASE_PATH = "src/artifacts/"
MODEL_FILENAME = "model.pkl"
MODEL_VERSION_PREFIX = "model_v"

metadata = {
    "accuracy": acc,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": rocauc,
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
