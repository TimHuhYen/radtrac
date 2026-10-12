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
    out = Path(__file__).parent.parent / "data"
    out.mkdir(exist_ok=True)
    path = out / "huntsville.csv"
    last_time = {} # icao24 -> time of last position saved

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for i in range(N_POLLS):
            try:
                r = requests.get(URL, params=BOX, timeout=15)
                if r.status_code == 429:
                    print("Out of credits for today, stopping. :(")
                    break
                r.raise_for_status()
                rows = parse_states(r.json())
            except requests.RequestException as e:
                print(f"poll {i}: request failed ({e}), skipping...")
                time.sleep(POLL_SECONDS)
                continue

            new = [row for row in rows if last_time.get(row["icao24"]) != row["time"]]
            for row in new:
                last_time[row["icao24"]] = row["time"]
                writer.writerow(row)
            f.flush()
            print(f"poll {i + 1}/{N_POLLS}: {len(rows)} planes in box, {len(new)} new positions")
            time.sleep(POLL_SECONDS)
    print(f"Saved {path}")

if __name__ == "__main__":
    main()