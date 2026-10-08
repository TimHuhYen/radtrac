import numpy as np
import pytest

from tracker.sensors import radar_measurement, radar_jacobian

def test_measurement_known_values():
    z = radar_measurement([3.0, 4.0, 0.0, 0.0])
    