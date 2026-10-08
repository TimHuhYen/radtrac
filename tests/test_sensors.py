import numpy as np
import pytest

from tracker.sensors import radar_measurement, radar_jacobian

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