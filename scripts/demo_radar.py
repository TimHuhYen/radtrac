from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tracker.ekf import radar_tracker
from tracker.sensors import radar_measurement, radar_jacobian, radar_to_xy
from tracker.simulate import rmse, simulate_radar_readings, simulate_track

DT = 1.0
ACCEL_STD = 8.0
RANGE_STD = 50.0
BEARING_STD = 0.002
RADAR = (-10000.0, -10000.0)

def main():
    truth, _ = simulate_track(dt=DT)
    readings = simulate_radar_readings(truth, RADAR, RANGE_STD, BEARING_STD)

    kf = radar_tracker(DT, ACCEL_STD, RANGE_STD, BEARING_STD, readings[0], RADAR)
    h = lambda x: radar_measurement(x, RADAR)
    jac = lambda x : radar_jacobian(x, RADAR)

    est = []
    for z in readings:
        kf.predict()
        kf.update_nonlinear(z, h, jac, angle_rows=(1,))
        est.append(kf.x.copy())
    est = np.array(est)

    raw_xy = np.array([radar_to_xy(z, RADAR) for z in readings])
    raw_err = rmse(raw_xy[10:], truth[10:, :2])
    filt_err = rmse(est[10:, :2], truth[10:, :2])
    print(f"Raw radar RMSE: {raw_err:6.1f} m")
    print(f"EKF RMSE: {filt_err:6.1f} m")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.scatter(raw_xy[:, 0], raw_xy[:, 1], s=6, color="0.7", label="Radar readings")
    ax1.plot(truth[:, 0], truth[:, 1], color="black", label="True path")
    ax1.plot(est[:, 0], est[:, 1], color="tab:red", label="EFL estimate")
    ax1.scatter(*RADAR, marker="^", s=80, color="tab:blue", label="Radar")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_aspect("equal")
    ax1.legend()

    t = np.arange(len(truth)) * DT
    ax2.plot(t, np.linalg.norm(raw_xy - truth[:, :2], axis=1),
             color="0.6", label=f"Radar readings (RMSE {raw_err:.0f} m)")
    ax2.plot(t, np.linalg.norm(est[:, :2] - truth[:, :2], axis=1),
             color="tab:red", label=f"EKF (RMSE {filt_err:.0f} m)")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("position error (m)")
    ax2.legend()

    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "radar_tracking.png", dpi=150, bbox_inches="tight")
    print("Saved results/radar_tracking.png")

if __name__ == "__main__":
    main()