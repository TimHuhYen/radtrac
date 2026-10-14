import numpy as np

from tracker.geo import latlon_to_xy
from tracker.sensors import radar_measurement

def resample(pts, grid, lat0, lon0):
    """
    Position (meters) of one plane at each time in grid.
    NaN where the plane isn't in the air
    """
    t = np.array([p[0] for p in pts])
    lat = np.array([p[1] for p in pts])
    lon = np.array([p[2] for p in pts])
    x, y = latlon_to_xy(np.interp(grid, t, lat), np.interp(grid, t, lon), lat0, lon0)
    outside = (grid < t[0]) | (grid > t[-1])
    x[outside] = np.nan
    y[outside] = np.nan
    return np.column_stack([x, y])

def make_scans(tracks, radar_pos, lat0, lon0, scan_seconds=10.0, range_std=50.0, bearing_std=0.001, p_detect=0.95, seed=0):
    """
    Turn real tracks into what one radar would see: unlabeled dots, one batch per scan.

    Returns (scans, truth). Each scan is a dict with
    "t":     time in sec
    "z":     array of [range, bearing]
    "owner": which plane made each readings. FOR SCORING ONLY ToT
    """
    t_start = min(pts[0][0] for pts in tracks.value())
    t_end = max(pts[-1][0] for pts in tracks.value())
    grid = np.arange(t_start, t_end, scan_seconds)
    ids = list(tracks)
    positions = np.array([resample(tracks[i], grid, lat0, lon0) for i in ids]) # planes x scans x 2

    rng = np.random.default_rng(seed)
    scans = []
    for k in range(len(grid)):
        readings, owners = [], []
        for n, icao in enumerate(ids):
            x, y = positions[n, k]
            if np.isnan(x) or rng.random() > p_detect:
                continue # not in the air, or the radar missed it this scan
            z = radar_measurement([x, y, 0.0, 0.0], radar_pos) + rng.normal(0.0, [range_std, bearing_std])
            readings.append(z)
            owners.append(icao)
        order = rng.permutation(len(readings))
        scans.append({
            "t": grid[k],
            "z": np.array(readings).reshape(-1, 2)[order],
            "owner": [owners[i] for i in order]
        })
    return scans, dict(zip(ids, positions))