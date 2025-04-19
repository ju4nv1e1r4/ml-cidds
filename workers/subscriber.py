import logging
import threading

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def health_check():
    return {"status": "ok"}


def start_subscriber():
    from workers.sub_new_data import main

    main()


@app.on_event("startup")
def startup_event():
    logging.info("Starting subscriber thread...")
    thread = threading.Thread(target=start_subscriber)
    thread.start()
