import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tracker.geo import latlon_to_xy

LAT0, LON0 = 34.6, -86.7 # center of box in fetch_opens.ypy

def load_tracks(path):
    tracks = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["on_ground"] == "True":
                continue
            tracks[row["icao24"]].append(
                (float(row["time"]), float(row["lat"]), float(row["lon"]), row["callsign"])
            )

    # have issue where last planes return array too low for .min()
    # proposed fix
    clean = {}
    for icao, pts in tracks.items():
        pts.sort() # oldest first
        kept = [pts[0]]
        for p in pts[1:]:
            if p[0] > kept[-1][0]: # only keep points where time moves forward
                kept.append(p)
        if len(kept) >= 3:
            clean[icao] = kept
    return clean

def main():
    root = Path(__file__).parent.parent
    tracks = load_tracks(root / "data" / "huntsville.csv")
    fig, ax = plt.subplots(figsize=(8, 8))
    for icao, pts in sorted(tracks.items(), key=lambda kv: -len(kv[1])):
        t = np.array([p[0] for p in pts])
        x, y = latlon_to_xy([p[1] for p in pts], [p[2] for p in pts], LAT0, LON0)
        gaps = np.diff(t)
        label = pts[0][3] or icao
        print(f"{icao} {label:8s} {len(pts):3d} points, gaps {gaps.min():.0f}-{gaps.max():.0f} s")
        ax.plot(x / 1000, y / 1000, marker=".", lw=1, label=label)
    ax.set_xlabel("east (km)")
    ax.set_ylabel("north (km)")
    ax.set_aspect("equal")
    ax.legend(fontsize=7)
    (root / "results").mkdir(exist_ok=True)
    fig.savefig(root / "results" / "flights.png", dpi=150, bbox_inches="tight")

if __name__ == "__main__":
    main()