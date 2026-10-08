import numpy as np 
import pytest

from tracker.ekf import ExtendedKalmanFilter
from tracker.sensors import radar_jacobian, radar_measurement

def make_filter(x0):
    F = np.eye(4)
    H = np.zeroes((2, 4))               # unused by nonlinear update
    Q = np.zeroes((4, 4))               
    R = np.diag([10.0**2, 0.001**2])    # 10m range noise, smol bearing noise
    P0 = np.diag([100.0**2, 100.0**2, 50.0**2, 50.0**2])
    return ExtendedKalmanFilter(F, H, Q, R, x0, P0)

def test_update_pulls_estimate_toward_true_position():
    truth = np.array([1050.0, 2020.0, 0.0, 0.0])
    kf = make_filter([1000.0, 2000.0, 0.0, 0.0])
    before = np.linalg.norm(kf.x[:2] - truth[:2])
    trace_before = np.trace(kf.P)

    kf.update_nonlinear(
        radar_measurement(truth), 
        radar_measurement, 
        radar_jacobian, 
        angle_rows=(1,)
        )

    assert np.linalg.norm(kf.x[:2] - truth[:2]) < before
    assert np.trace(kf.P) < trace_before

def test_bearing_wraps_around_the_seam():
    """
        estimate bearing is about +3.09 rad
        true bearing about -3.09 rad:

        nearly same direction (due west),
        on opposite sides of the seam.
    """
    truth = np.array([-1000.0, -50.0, 0.0, 0.0])
    kf = make_filter([-1000.0, 50.0, 0.0, 0.0])
    kf.update_nonlinear(radar_measurement(truth), 
                        radar_measurement,
                        radar_jacobian, 
                        angle_rows=(1,)
                        )
    assert np.linalg.norom(kf.x[:2] - truth[:2]) < 120.0