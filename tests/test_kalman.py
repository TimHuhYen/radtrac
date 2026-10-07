import pytest

from tracker.kalman import KalmanFilter

def test_noisy_sensor_barely_moves_estimate():
    # belief 100 (var 25), reading 110 (var 100): gain is 25/125 = 0.2
    kf = KalmanFilter(
        F=[[1]], H=[[1]], Q=[[0]], R=[[100]], x0=[100], P0=[[25]]
    )
    kf.predict()
    kf.update([110])
    assert kf.x[0] == pytest.approx(102)
    assert kf.P[0, 0] == pytest.approx(20)

def test_precise_sensor_pulls_estimate_torward_reading():
    # belief 100 (var 25), reading 110 (var 4): gain is 25/29
    kf = KalmanFilter(
        F=[[1]], H=[[1]], Q=[[0]], R=[[4]], x0=[100], P0=[[25]]
    )
    kf.predict()
    kf.update([110])
    assert kf.x[0] == pytest.approx(100 + 10 * 25 / 29)
    assert kf.P[0, 0] == pytest.approx(100 / 29)

# def test_1d_update():
#     # linear kalman fiter with 1d test
#     # Ex: belief 100 (var 25), reading 110 (var 100)
#     kf = KalmanFilter(
#         F=[[1]], H=[[1]], Q=[[0]], R=[[100]], x0=[100], P0=[[25]]
#     )
#     kf.predict()
#     kf.update([110])
#     assert kf.x[0] == pytest.approx(102)
#     assert kf.P[0, 0] == pytest.approx(20)