from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tracker.kalman import constant_velocity_2d
from tracker.simulate import rmse, simulate_track

DT = 1.0
MEAS_STD = 50.0
ACCEL_STD = 8.0

def main():
    truth, readings = simulate_track(dt=DT, meas_std=MEAS_STD)
    kf = constant_velocity_2d(DT, ACCEL_STD, MEAS_STD, readings[0])

    est = []
    for z in readings:
        kf.predict()
        kf.update(z)
        est.append(kf.x.copy())
    est = np.array(est)

    raw_err = rmse(readings[10:], truth[10:, :2])
    filt_err = rmse(est[10:, :2], truth[10:, :2])
    print(f"Raw error: {raw_err:6.1f} m")
    print(f"Filtered error: {filt_err:6.1f} m")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.scatter(readings[:, 0], readings[:, 1], s=6, color="0.7", label="Readings")
    ax1.plot(truth[:, 0], truth[:, 1], color="black", label="Truth path")
    ax1.plot(est[:, 0], est[:, 1], color="tab:red", label="Kalman estimate")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_aspect("equal")
    ax1.legend()

    t = np.arange(len(truth)) * DT
    ax2.plot(t, np.linalg.norm(readings - truth[:, :2], axis=1),
                color="0.6", label=f"Readings (RMSE {raw_err:.0f} m)")
    ax2.plot(t, np.linalg.norm(est[:, :2] - truth[:, :2], axis=1),
                color="tab:red", label=f"Kalman (RMSE {filt_err:.0f} m)")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("position error (m)")
    ax2.legend()

    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "tracking.png", dpi=150, bbox_inches="tight")
    print(f"Saved results/tracking.png")

if __name__ == "__main__":
    main()