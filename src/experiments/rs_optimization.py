import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from google.resumable_media.common import InvalidResponse
import logging
from colorama import Fore, init
import io

from src.ml.optmize import Optimize
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
    print("Tipo de load:", type(load))
    print("Conteúdo bruto:", load[:100])
    print("---")
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
        + f"Data successfully splitted -> Train: {X_train.shape} || Test: {X_test.shape}"
    )
except Exception as split_error:
    logging.error(Fore.RED + f"Error: {split_error}")

print("---\n")
print(
    Fore.YELLOW + "Starting hyperparameter optimization with RandomForestClassifier..."
)


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_grid = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 10, 20, 30, 50],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
    "bootstrap": [True, False],
    "criterion": ["gini", "entropy"],
}

optimize_rf = Optimize(
    RandomForestClassifier(),
    rf_grid,
    X_train,
    X_test,
    y_train_shuffled,
    y_test,
    cv=skf,
    scoring="roc_auc",
)

rf = optimize_rf.with_random_search()

print(Fore.GREEN + "1/2 - Optimization: RandomizedSearchCV + RandomForestClassifier")
print("---\n")
print(Fore.YELLOW + "Starting hyperparameter optimization with com XGBClassifier...")

xgb_grid = {
    "n_estimators": [100, 200, 300, 500],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 5, 7, 10],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "gamma": [0, 0.1, 0.2, 0.3],
    "reg_alpha": [0, 0.01, 0.1, 1],
    "reg_lambda": [1, 1.5, 2],
}

optimize_xgb = Optimize(
    XGBClassifier(),
    xgb_grid,
    X_train,
    X_test,
    y_train_shuffled,
    y_test,
    cv=skf,
    scoring="roc_auc",
)

xgb = optimize_xgb.with_random_search()
print(Fore.GREEN + "2/2 - Optimization: RandomizedSearchCV + XGBClassifier")

try:
    rf_sanity = RandomForestClassifier(**rf)
    rf_sanity.fit(X_train, y_train_shuffled)
    score = rf_sanity.score(X_test, y_test)
    logging.info(f"Sanity check accuracy with shuffled target: {score:.4f}")
except Exception as sanity_check_error:
    logging.error(f"Sanity Check Error: {sanity_check_error}")
print("End of optimization")
