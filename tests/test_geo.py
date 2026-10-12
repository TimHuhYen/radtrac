import pytest

from tracker.geo import latlon_to_xy

def test_reference_point_is_origin():
    x, y = latlon_to_xy(34.6, -86.6, 34.6, -86.6)
    assert x == pytest.approx(0.0)
    assert y == pytest.approx(0.0)

def test_one_degree_noth_is_about_111_km():
    x, y = latlon_to_xy(35.6, -86.6, 34.6, -86.6)
    assert x == pytest.approx(0.0)
    assert y == pytest.approx(111195, rel=1e-3)

def test_east_west_degrees_shrink_with_latitude():
    x_eq, _ = latlon_to_xy(0.0, 1.0, 0.0, 0.0)
    x_60, _ = latlon_to_xy(60.0, 1.0, 60.0, 0.0)
    assert x_60 == pytest.approx(x_eq / 2, rel=1e-3)