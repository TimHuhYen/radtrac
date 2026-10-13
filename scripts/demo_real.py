import sys
from functools import partial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.plot_flights import LAT0, LON0, load_tracks
from tracker.ekf import radar_tracker
from tracker.geo import latlon_to_xy
from tracker.kalman import cv_matrices
from tracker.sensors import radar_to_xy, radar_jacobian, radar_measurement
from tracker.simulate import rmse, simulate_radar_readings

RANGE_STD = 50.0
BEARING_STD = 0.002
ACCEL_STDS = [0.5, 2.0, 5.0, 10.0] # m/s^2 values to try

def run_tracker(t, radars, all_readings, accel_std):
    """
    Track with time gaps, t holds time of each reading in seconds
    """
    kf = radar_tracker(1.0, accel_std, RANGE_STD, BEARING_STD, all_readings[0][0], radars[0])
    est = [kf.x.copy()]
    for k in range(1, len(t)):
        kf.predict(*cv_matrices(t[k] - t[k - 1], accel_std))
        for pos, readings in zip(radars, all_readings):
            kf.update_nonlinear(
                readings[k],
                partial(radar_measurement, radar_pos=pos),
                partial(radar_jacobian, radar_pos=pos),
                angle_rows=(1,)
            )
        est.append(kf.x.copy())
    return np.array(est)

def main():
    icao = sys.argv[1] if len(sys.argv) > 1 else "a1aaaa"
    root = Path(__file__).parent.parent
    pts = load_tracks(root / "data" / "huntsville.csv")[icao]
    label = pts[0][3] or icao

    t = np.array([p[0] for p in pts])
    t = t - t[0]
    x, y = latlon_to_xy([p[1] for o in pts], [p[2] for p in pts], LAT0, LON0)
    truth = np.column_stack([x, y, np.zeros_like(x), np.zeros_like(x)])

    # Two virtual radars, 30 mk south-west and south-east of the middle of track
    cx, cy = x.mean(), y.mean()
    radars = [(cx - 30000.0, cy - 30000.0), (cx + 30000.0, cy - 30000.0)]
    reads_a = simulate_radar_readings(truth, radars[0], RANGE_STD, BEARING_STD, seed=1)
    reads_b = simulate_radar_readings(truth, radars[1], RANGE_STD, BEARING_STD, seed=2)

    s = 3 # skip first few points while filter locks on
    raw_a = np.array([radar_to_xy(z, radars[0]) for z in reads_a])
    raw_b = np.array([radar_to_xy(z, radars[1]) for z in reads_b])
    print(f"{label}: {len(t)} points over {t[-1]:.0f} s")
    print(f"Raw radar A: {rmse(raw_a[s:], truth[s:, :2]):6.1f} m")
    print(f"Raw radar B: {rmse(raw_b[s:], truth[s:, :2]):6.1f} m")

    best = None
    for a in ACCEL_STDS:
        est = run_tracker(t, radars, [reads_a, reads_b], a)
        err = rmse(est[s:, :2], truth[s:, :2])
        print(f"Fused, accel_std {a:4.1f}: {err:6.1f} m")
        if best is None or err < best[0]:
            best = (err, a, est)
    err, a, est = best

    