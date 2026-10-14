from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.plot_flights import LAT0, LON0, load_tracks
from tracker.multi import MultiTracker
from tracker.scenario import make_scans

RADAR = (0.0, -60000.0)
RANGE_STD = 50.0
BEARING_STD = 0.001
ACCEL_STD = 2.0
MIN_POINTS = 10

def main():
    root = Path(__file__).parent.parent
    tracks = {k: v for k, v in load_tracks(root / "data" / "huntsville.csv").items()
              if len(v) >= MIN_POINTS}
    scans, truth = make_scans(tracks, RADAR, LAT0, LON0,
                              range_std=RANGE_STD, bearing_std=BEARING_STD)

    mt = MultiTracker(RADAR, RANGE_STD, BEARING_STD, ACCEL_STD)
    updates = defaultdict(int)       # track id -> dots it received
    owners_of = defaultdict(list)   # track id -> which real plane each dot came from (grading only)
    planes_of = defaultdict(set)    # real plane -> track ids that ever followed it
    paths = defaultdict(list)       # track id -> estimated positions
    switches = []                   # (x, y) where a track jumped from one plan to anotha
    n_planes, n_tracks = [], []

    for s in scans:
        used = mt.step(s["t"], s["z"])
        for tid, j in used:
            owner = s["owner"][j]
            if owners_of[tid] and owners_of[tid][-1] != owner:
                live = next(tr for tr in mt.tracks if tr.id == tid)
                switches.append((live.kf.x[0], live.kf.x[1]))
            owners_of[tid].append(owner)
            planes_of[owner].add(tid)
            updates[tid] += 1
        for tr in mt.tracks:
            paths[tr.id].append((tr.kf.x[0], tr.kf.x[1]))
        n_planes.append(len(s["z"]))
        n_tracks.append(sum(mt.confirmed(tr) for tr in mt.tracks))

    real = [tid for tid, n in updates.items() if n >= mt.confirm_hits]
    extra = sum(len(ids) - 1 for ids in planes_of.values())
    print(f"{len(tracks)} plabes, {len(scans)} scans")
    print(f"tracks started: {len(updates)}, reached confirmed: {len(real)}")
    print(f"ID switches (a track jumped to a different plane): {len(switches)}")
    print(f"extra tracks (a plane followed by more than one track): {extra}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1]})
    for tid in real:
        p = np.array(paths[tid]) / 1000
        ax1.plot(p[:, 0], p[:, 1], lw=1.2)
    if switches:
        sw = np.array(switches) / 1000
        ax1.scatter(sw[:, 0], sw[:, 1], marker="x", s=70, color="red", zorder=5, label="ID switch")
        ax1.legend()
    ax1.set_title("tracks (one color each)")
    ax1.set_xlabel("east (km)")
    ax1.set_ylabel("north (km)")
    ax1.set_aspect("equal")

    t = [s["t"] - scans[0]["t"] for s in scans]
    ax2.plot(t, n_planes, color="0.5", label="dots in scan")
    ax2.plot(t, n_tracks, color="tab:red", label="confirmed tracks")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("count")
    ax2.legend()

    (root / "results").mkdir(exist_ok=True)
    fig.savefig(root / "results" / "multi_tracking.png", dpi=150, bbox_inches="tight")
    print("Saved results/multi_tracking.png")

if __name__ == "__main__":
    main()