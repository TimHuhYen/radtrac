from functools import partial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tracker.ekf import radar_tracker
from tracker.sensors import radar_to_xy, radar_jacobian, radar_measurement
from tracker.simulate import rmse, simulate_radar_readings, simulate_track

DT = 1.0
ACCEL_STD = 8.0
RANGE_STD = 50.0
BEARING_STD = 0.002
RADAR_A = (-10000.0, -10000.0)
RADAR_B = (40000.0, -10000.0)

def run_tracker(radars, all_readings):
    """
        Track with one or more radars.
        all_readings[i] is radars[i]
    """
    kf = radar_tracker(
        DT, ACCEL_STD,
        RANGE_STD, BEARING_STD, 
        all_readings[0][0], radars[0])
    est = []
    for k in range(len(all_readings[0])):
        kf.predict()
        for pos, readings in zip(radars, all_readings):
            kf.update_nonlinear(
                readings[k],
                partial(radar_measurement, radar_pos=pos),
                partial(radar_jacobian, radar_pos=pos),
                angle_rows=(1,),
            )
        est.append(kf.x.copy())
    return np.array(est)

def main():
    truth, _ = simulate_track(dt=DT)
    reads_a = simulate_radar_readings(truth, RADAR_A, RANGE_STD, BEARING_STD, seed=1)
    reads_b = simulate_radar_readings(truth, RADAR_B, RANGE_STD, BEARING_STD, seed=2)

    est_a = run_tracker([RADAR_A], [reads_a])
    est_b = run_tracker([RADAR_B], [reads_b])
    est_ab = run_tracker([RADAR_A, RADAR_B], [reads_a, reads_b])

    def err(est):
        return rmse(est[10:, :2], truth[10:, :2])

    raw_a = np.array([radar_to_xy(z, RADAR_A) for z in reads_a])
    raw_b = np.array([radar_to_xy(z, RADAR_B) for z in reads_b])
    print(f"")
    print(f"")
    print(f"")
    print(f"")
    print(f"")

    