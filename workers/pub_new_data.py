import json
import os

import dotenv
from google.cloud import pubsub_v1

dotenv.load_dotenv()

publisher = pubsub_v1.PublisherClient()

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
TOPIC_ID = os.getenv("GCP_PUBSUB_TOPIC")

TOPIC_PATH = publisher.topic_path(PROJECT_ID, TOPIC_ID)


def publish_new_data(data: dict):
    message_json = json.dumps(data)
    message_bytes = message_json.encode("utf-8")

    future = publisher.publish(TOPIC_PATH, data=message_bytes)
    return future
