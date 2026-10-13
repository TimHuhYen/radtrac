import pytest
import numpy as np

from tracker.kalman import KalmanFilter, constant_velocity_2d, cv_matrices

"""
    UNADJUSTED FOR NEW F, Q FOR NON-FIXED DT !!!
"""

# def test_plane_predict_moves_state_forward():
#     kf = constant_velocity_2d(dt=2.0, accel_std=1.0, meas_std=10.0, z0=[0, 0])
#     kf.x = np.array([0.0, 0.0, 100.0, -50.0])
#     kf.predict()
#     assert kf.x == pytest.approx([200.0, -100.0, 100.0, -50.0])

# def test_plane_velocity_learned_from_positions_only():
#     rng = np.random.default_rng(3)
#     t = np.arange(200)
#     truth = np.column_stack([100.0 * t, 50.0 * t])
#     readings = truth + rng.normal(0, 20.0, size=truth.shape)
#     kf = constant_velocity_2d(dt=1.0, accel_std=0.1, meas_std=20.0, z0=readings[0])
#     for z in readings[1:]:
#         kf.predict()
#         kf.update(z)
#     assert kf.x[2:] == pytest.approx([100.0, 50.0], abs=5.0)

def test_noisy_sensor_barely_moves_estimate():
    # belief 100 (var 25), reading 110 (var 100): gain is 25/125 = 0.2
    kf = KalmanFilter(
        F=[[1]], H=[[1]], Q=[[0]], R=[[100]], x0=[100], P0=[[25]]
    )
    kf.predict()
    kf.update([110])
    assert kf.x[0] == pytest.approx(102)
    assert kf.P[0, 0] == pytest.approx(20)

# def test_precise_sensor_pulls_estimate_torward_reading():
#     # belief 100 (var 25), reading 110 (var 4): gain is 25/29
#     kf = KalmanFilter(
#         F=[[1]], H=[[1]], Q=[[0]], R=[[4]], x0=[100], P0=[[25]]
#     )
#     kf.predict()
#     kf.update([110])
#     assert kf.x[0] == pytest.approx(100 + 10 * 25 / 29)
#     assert kf.P[0, 0] == pytest.approx(100 / 29)

def test_predict_with_a_different_time_step():
    kf = constant_velocity_2d(dt=1.0, accel_std=1.0, meas_std=10.0, z0=[0, 0])
    kf.x = np.array([0.0, 0.0, 100.0, -50.0])
    F, Q = cv_matrices(dt=5.0, accel_std=1.0)
    kf.predict(F, Q)
    assert np.allclose(kf.x, [500.0, -250.0, 100.0, -50.0])

def test_longer_gap_means_more_doubt():
    short = constant_velocity_2d(1.0, 10., 10.0, [0, 0])
    long_ = constant_velocity_2d(1.0, 1.0, 10.0, [0, 0])
    short.predict(*cv_matrices(1.0, 1.0))
    long_.predict(*cv_matrices(20.0, 1.0))