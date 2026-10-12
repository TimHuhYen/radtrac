import csv
import time
from pathlib import Path

import requests

URL = "https://opensky-network.org/api/states/all"
BOX = {"lamin": 34.1, "lamax": 35.1, "lomin": -87.3, "lomax": -86.3}    # around Huntsville, AL
POLL_SECONDS = 10
N_POLLS = 90 # like, 15 minutes
COLUMNS = ["icao24", "callsign", "time", "lat", "lon", "altitude", "speed", "heading", "on_ground"]

