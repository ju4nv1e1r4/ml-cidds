import os
import json
import pandas as pd
from google.cloud import pubsub_v1
from utils.gcp import CloudStorageOps
import dotenv
import logging

dotenv.load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
SUBSCRIPTION_ID = os.getenv("GCP_PUBSUB_SUBSCRIPTION")
subscriber = pubsub_v1.SubscriberClient()

SUBSCRIPTION_PATH = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_ID)


def callback(message: pubsub_v1.subscriber.message.Message) -> None:
    try:
        logging.info("Received message.")
        data = json.loads(message.data.decode("utf-8"))

        mode = data["mode"]
        input_data = data["data"]
        prediction = data["prediction"]

        input_data["prediction"] = prediction
        df = pd.DataFrame([input_data])

        filename = "new_data_sup.csv" if mode == "supervised" else "new_data_unsup.csv"
        local_path = f"data/{filename}"
        remote_path = f"src/data/{filename}"

        file_exists = os.path.exists(local_path)
        df.to_csv(local_path, mode="a", header=not file_exists, index=False)

        gcs = CloudStorageOps("ml-anomaly-detection")
        gcs.upload_to_bucket(local_path, remote_path)

        logging.info(f"Arquivo salvo e enviado para o bucket: {remote_path}")

        message.ack()
    except Exception as publish_error:
        logging.error(f"Error processing message: {publish_error}")
        message.nack()


def main():
    logging.info("Starting subscriber...")
    streaming_pull_future = subscriber.subscribe(SUBSCRIPTION_PATH, callback=callback)
    logging.info(f"Listening for messages on {SUBSCRIPTION_PATH}...\n")

    try:
        streaming_pull_future.result()
    except KeyboardInterrupt:
        streaming_pull_future.cancel()
        logging.info("Subscriber stopped.")


if __name__ == "__main__":
    main()
