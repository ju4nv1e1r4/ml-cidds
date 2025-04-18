from pydantic import BaseModel
from typing import Literal
from datetime import datetime


class SupervisedSessionData(BaseModel):
    start_session: str
    end_session: str
    packets: int
    bytes: int
    source_port: float
    flag: str


class UnsupervisedSessionData(BaseModel):
    start_session: str
    end_session: str
    packets: int
    bytes: int
    source_port: float
    flag: str
    source_ip_freq: float
    dest_ip_freq: float
    network_protocol: Literal["TCP", "UDP", "ICMP"]
