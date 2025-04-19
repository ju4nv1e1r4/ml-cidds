import datetime
import json
import logging
import os
import pickle
import time

from colorama import Fore, init
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
from sklearn.model_selection import StratifiedKFold

from src.ml.optmize import Optimize
from src.models.load import Load
from utils.constants import Config
from utils.gcp import CloudStorageOps

init(autoreset=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
c = Config()
logger = c.logger
gcs = CloudStorageOps("ml-anomaly-detection")

PATH_FILE = "preprocessed/CIDDS-001.csv"

data = Load(file_path=PATH_FILE)

df = data.load_dataset()
X_train, X_test, y_train, y_test = data.split_dataset(df)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 20, None],
    "min_samples_split": [2, 5, 10],
}

my_thresholds = {
    "recall": 0.90,
    "precision": 0.60,
    "roc_auc": 0.40,
    "f1_score": 0.75,
    "accuracy": 0.5,
}

opt = Optimize(
    RandomForestClassifier(),
    param_grid,
    X_train,
    X_test,
    y_train,
    y_test,
    cv=skf,
    verbose=0,
    n_iter=1,
    scoring="recall",
    thresholds=my_thresholds,
)

logging.info("===== Randomized Search =====")
best_params_random = opt.with_random_search()

logging.info("\n===== Bayesian Search =====")
best_params_bayes = opt.with_bayesian_search()

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
    model.fit(X_train, y_train)
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
    "f1_score": f1,
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
    "features": list(X_train.columns),
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
