import csv
import time
from pathlib import Path

import requests

URL = "https://opensky-network.org/api/states/all"
BOX = {"lamin": 34.1, "lamax": 35.1, "lomin": -87.3, "lomax": -86.3}    # around Huntsville, AL
POLL_SECONDS = 10
N_POLLS = 90 # like, 15 minutes
COLUMNS = ["icao24", "callsign", "time", "lat", "lon", "altitude", "speed", "heading", "on_ground"]

def parse_states(payload):
    """
    Turn the APIs list of lists into directories
    Skip planes with no position
    """
    rows = []
    for s in payload.get("states") or []:
        if s[3] is None or s[5] is None or s[6] is None:
            continue
        rows.append({
            "icao24": s[0],
            "callsign": (s[1] or "").strip(),
            "time": s[3],
            "lat": s[6],
            "lon": s[5],
            "altitude": s[7],
            "speed": s[9],
            "heading": s[10],
            "on_ground": s[8]
        })
    return rows

def main():
    pass