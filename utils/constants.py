import structlog
import logging
import json
import os
from google.auth import jwt
from google.cloud import pubsub_v1
from logging.handlers import RotatingFileHandler


class Config:
    def __init__(self):
        self.logger = self.init_logging()

    def init_logging(self):
        log_handler = RotatingFileHandler(
            "data/logs/app.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )

        logging.basicConfig(handlers=[log_handler], level=logging.INFO)

        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.JSONRenderer(),
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        )

        return structlog.get_logger()

    def gcp_auth(self):
        service_account_file = os.environ["GCP_SERVICE_ACCOUNT_FILE"]
        service_account_info = json.loads(open(service_account_file).read())

        sub_audience = "https://pubsub.googleapis.com/google.pubsub.v1.Subscriber"
        pub_audience = "https://pubsub.googleapis.com/google.pubsub.v1.Publisher"

        credentials_pub = jwt.Credentials.from_service_account_info(
            service_account_info,
            audience=pub_audience,
        )

        credentials_sub = jwt.Credentials.from_service_account_info(
            service_account_info,
            audience=sub_audience,
        )

        subscriber = credentials_sub.with_claims(audience=pub_audience)

        publisher = credentials_pub.with_claims(audience=sub_audience)

        pub = pubsub_v1.PublisherClient(credentials=publisher)
        sub = pubsub_v1.SubscriberClient(credentials=subscriber)

        return pub, sub
