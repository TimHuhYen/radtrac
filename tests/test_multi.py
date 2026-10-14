import numpy as np
from tracker.multi import MultiTracker
from tracker.sensors import radar_measurement

RADAR = (0.0, -60000.0)

def dot(x, y):
    return radar_measurement([x, y, 0.0, 0.0], RADAR)

def make_tracker():
    return MultiTracker(RADAR, range_std=50.0, bearing_std=0.001, accel_std=2.0)

def test_one_plane_becomes_one_confrmed_tracl():
    mt = make_tracker()
    for k in range(10):
        mt.step(10.0 * k, np.array([dot(-5000.0 + 1000.0 * k, 0.0)])) # 100 m/s east
    assert len(mt.tracks) == 1
    assert mt.confirmed(mt.tracks[0])
    assert np.allclose(mt.tracks[0].kf.x[:2], [4000.0, 0.0], atol=100.0)

def test_two_far_apart_planes_keep_separate_identities():
    mt = make_tracker()
    ids = []
    for k in range(10):
        a = dot(-30000.0 + 1000.0 * k, 0.0)
        b = dot(30000.0 - 1000.0 * k, 20000.0)
        ids.append(mt.step(10.0 * k, np.array([a, b])))
    assert len(mt.tracks) == 2
    assert all(mt.confirmed(t) for t in mt.tracks)
    first = dict((j, tid) for tid, j in ids[0])     # dot index -> track id at scan 0
    for pairs in ids:
        assert dict((j, tid) for tid, j in pairs) == first

def test_dot_far_from_every_track_starts_a_new_track():
    mt = make_tracker()
    for k in range(5):
        mt.step(10.0 * k, np.array([dot(-5000.0 + 1000.0 * k, 0.0)]))
    mt.step(50.0, np.array([dot(0.0, 0.0), dot(40000.0, 30000.0)]))
    assert len(mt.tracks) == 2

def test_track_is_dropped_after_too_many_misses():
    mt = make_tracker()
    for k in range(5):
        mt.step(10.0 * k, np.array([dot(-5000.0 + 1000.0 * k, 0.0)]))
    assert len(mt.tracks) == 1
    for k in range(5, 5 + mt.max_misses):
        mt.step(10.0 * k, np.empty((0, 2)))
        assert len(mt.tracks) == 1      # still costing
    mt.step(10.0 * (5 + mt.max_misses), np.empty((0, 2)))
    assert len(mt.tracks) == 0