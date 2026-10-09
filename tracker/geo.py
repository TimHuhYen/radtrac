import numpy as np

EARTH_RADIUS = 6371000.0 # meters btw

def latlon_to_xy(lat, lon, lat0, lon0):
    """
    Convert deg of lat/lon to meters east(x) and north(y)
        - of reference point
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    x = np.asarray(lon - lon0) * np.cos(np.radians(lat0)) * EARTH_RADIUS
    y = np.asarray(lat - lat0) * EARTH_RADIUS
    return x, y