import numpy as np
import pytest

from tracker.sensors import radar_measurement, radar_jacobian, radar_to_xy
from tracker.scenario import make_scans, resample


def test_measurement_known_values():
    z = radar_measurement([3.0, 4.0, 0.0, 0.0])
    assert z == pytest.approx([5.0, np.arctan2(4.0, 3.0)])

def test_jacobian_known_values():
    J = radar_jacobian([3.0, 4.0, 0.0, 0.0])
    expected = [[0.6, 0.8, 0.0, 0.0],
                [-0.16, 0.12, 0.0, 0.0]]
    assert J == pytest.approx(np.array(expected))

def test_radar_position_shifts_the_view():
    # plane 3 east and 4 north of radar at (100, 50): same as origin case
    z = radar_measurement([103.0, 54.0, 0.0, 0.0], radar_pos=(100.0, 50.0))
    assert z == pytest.approx([5.0, np.arctan2(4.0, 3.0)])

def test_jacobian_matches_numeric_nudge():
    # nudge each state number a tiny bit each time and measure actual slope
    state = np.array([300.0, -400.0, 10.0, 5.0])
    eps = 1e-5
    J = radar_jacobian(state)
    for i in range(4):
        up = state.copy()
        down = state.copy()
        up[i] += eps
        down[i] -= eps
        slope = (radar_measurement(up) - radar_measurement(down)) / (2 * eps)
        assert J[:, i] == pytest.approx(slope, abs=1e-6)

def test_radar_to_xy_undo_measurements():
    state= np.array([5000.0, -2000.0, 0.0, 0.0])
    radar = (-1000.0, 3000.0)
    xy = radar_to_xy(radar_measurement(state, radar), radar)
    assert xy == pytest.approx(state[:2])

def straight_plane(t0, t1, lat_start=34.6):
    """
    A plane flying due north, reporting every 10 s from t0 to t1.
    """
    times = np.arange(t0, t1 + 1, 10.0)
    return [(t, lat_start + 0.0009 * (t - t0), -86.7, "TEST") for t in times]

def test_resample_interpolates_and_blanks_outside():
    pts = straight_plane(100, 200)
    grid = np.array([50.0, 105.0, 200.0, 250.0])
    xy = resample(pts, grid, 34.6, -86.7)
    assert np.all(np.isnan(xy[0])) and np.all(np.isnan(xy[3]))          # before take off, after landing
    assert xy [1, 1] == pytest.approx(0.0009 * 5 * 111195, rel=1e-2)    # 5 s after the first report
    assert xy[2, 1] == pytest.approx(0.0009 * 100 * 111195, rel=1e-2)

def test_scans_have_one_unlabeled_dot_per_plane_present():
    tracks = {"a": straight_plane(0, 100), "b": straight_plane(50, 150, lat_start=34.7)}
    scans, truth = make_scans(tracks, (0.0, -60000.0), 34.6, -86.7, range_std=1e-9, bearing_std=1e-9, p_detect=1.0)

    counts = [len(s["z"]) for s in scans]
    assert counts[0] == 1 and max(counts) == 2 # b has not appeared yet at the start
    # with no noise, every dat londs exaclt on its owner's true position >:)
    for s in scans:
        for z, owner in zip(s["z"], s["ower"]):
            k = int(round((s["t"] - scans[0]["t"]) / 10.0))
            assert radar_to_xy(z, (0.0, -60000.0)) == pytest.approx(truth[owner][k], abs=1e-3)

def test_missed_detections_happen():
    tracks = {"a": straight_plane(0, 1000)}
    scans, _ = make_scans(tracks, (0.0, -60000.0), 34.6, -86.7, p_detect=0.5)
    counts = [len(s["z"]) for s in scans]
    assert 0 in counts and 1 in counts