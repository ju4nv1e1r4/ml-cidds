import structlog
import logging
from logging.handlers import RotatingFileHandler

class Config:
    def __init__(self):
        self.logger = self.init_logging()

    def init_logging(self):
        log_handler = RotatingFileHandler(
            "data/logs/app.log", maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
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