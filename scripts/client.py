import requests
import random
import datetime
import schedule
import time
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

API_URL = (
    "https://anomaly-detection-api-1049636244984.us-central1.run.app/detect_anomaly"
)
COMMON_PORTS = [80, 443, 22, 21]
PROTOCOLS = ["TCP", "UDP", "ICMP"]
FLAGS = ["SYN", "ACK", "RST", "FIN", ""]


def generate_supervised_data():
    now = datetime.datetime.now()
    duration_seconds = random.randint(1, 10)
    end_time = now + datetime.timedelta(seconds=duration_seconds)

    packets = random.randint(100, 5000)
    bytes_flow = random.randint(10**5, 10**8)
    source_port = random.choice(COMMON_PORTS + [random.randint(1024, 65535)])
    flag = random.choice(FLAGS)

    data = {
        "mode": "supervised",
        "supervised_data": {
            "start_session": now.isoformat(),
            "end_session": end_time.isoformat(),
            "packets": packets,
            "bytes": bytes_flow,
            "source_port": float(source_port),
            "flag": flag,
        },
    }
    return data


def generate_unsupervised_data():
    now = datetime.datetime.now()
    duration_seconds = random.randint(1, 15)
    end_time = now + datetime.timedelta(seconds=duration_seconds)

    packets = random.randint(1, 5000)
    bytes_flow = random.randint(10**2, 10**8)
    source_port = random.choice(COMMON_PORTS + [random.randint(1024, 65535)])
    flag = random.choice(FLAGS)
    protocol = random.choice(PROTOCOLS)

    data = {
        "mode": "unsupervised",
        "unsupervised_data": {
            "start_session": now.isoformat(),
            "end_session": end_time.isoformat(),
            "packets": packets,
            "bytes": bytes_flow,
            "source_port": float(source_port),
            "flag": flag,
            "source_ip_freq": round(random.uniform(0.0001, 0.1), 4),
            "dest_ip_freq": round(random.uniform(0.0001, 0.1), 4),
            "network_protocol": protocol,
        },
    }
    return data


def send_request(data):
    try:
        response = requests.post(API_URL, json=data)
        logging.info(f"Status: {response.status_code} - Response: {response.text}")
    except Exception as e:
        logging.error(f"[ERROR] {e}")


def task_supervised():
    data = generate_supervised_data()
    send_request(data)


def task_unsupervised():
    data = generate_unsupervised_data()
    send_request(data)


schedule.every(2).minutes.do(task_supervised)
schedule.every(4).minutes.do(task_unsupervised)

logging.info("Client is working hard... Press Ctrl+C if you want to stop.")
while True:
    schedule.run_pending()
    time.sleep(1)
