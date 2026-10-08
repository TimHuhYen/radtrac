from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tracker.ekf import radar_tracker
from tracker.sensors import radar_measurement, radar_jacobian, radar_to_xy
from tracker.simulate import rmse, simulate_radar_readings, simulate_track

DT = 10.
ACCEL_STD = 8.0
RANGE_STD = 50.0
BEARING_STD = 0.002
RADAR = (-10000.0, -10000.0)

def main():
    truth, _ = simulate_track(dt=DT)
    readings = simulate_radar_readings(truth, RADAR, RANGE_STD, BEARING_STD)

    kf = simulate_radar_readings(DT, ACCEL_STD, RANGE_STD, BEARING_STD, readings[0], RADAR)
    h = lambda x: radar_measurement(x, RADAR)
    jac = lambda x : radar_jacobian(x, RADAR)

    est = []
    for z in readings:
        kf.predict()
        kf.update_nonlinear(z, h, jac, angle_rows=(1,))
        est.append(kf.x.copy())
    est = np.array(est)