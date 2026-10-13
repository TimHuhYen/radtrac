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