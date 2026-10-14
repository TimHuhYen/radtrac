from functools import partial

import numpy as np
from scipy.optimize import linear_sum_assignment

from tracker.ekf import radar_tracker
from tracker.kalman import cv_matrices
from tracker.sensors import radar_jacobian, radar_measurement

BIG = 1e6 # cost for a pairing that fails the gate

class Track:
    def __init__(self, track_id, kf, t):
        self.id = track_id
        self.kf = kf
        self.hits = 1
        self.misses = 0
        self.path = [(t, kf.x[0], kf.x[1])]

class MultiTracker:
    """
    
    """
    def __init__(self, radar_pos, range_std, bearing_std, accel_std,
                 gate=9.21, confirm_hits=3, max_misses=3, vel_std=150.0):
        self.radar_pos = radar_pos
        self.range_std = range_std
        self.bearing_std = bearing_std
        self.accel_std = accel_std
        self.gate = gate
        self.confirm_hits = confirm_hits
        self.max_misses = max_misses
        self.vel_std = vel_std
        self.tracks =[]
        self.next_id = 0
        self.t = None

    def confirmed(self, track):
        return track.hits >= self.confirm_hits

    def _distances(self, track, Z):
        """
        
        """
        x, P = track.kf.x, track.kf.P
        h = radar_measurement(x, self.radar_pos)
        H = radar_jacobian(x, self.radar_pos)
        R = np.diag([self.range_std**2, self.bearing_std**2])
        S_inv = np.linalg.inv(H @ P @ H.T + R)
        Y = Z - h
        Y[:, 1] = (Y[:, 1] + np.pi) % (2 * np.pi) - np.pi # bearing wraps round
        return np.einsum("ij,jk,ik->i", Y, S_inv, Y)

    def _match(self, tracks, free, Z):
        """
        
        """
        if not tracks or not free:
            return [], list(tracks)
        cost = np.array([self._distances(tr, Z[free]) for tr in tracks])
        rows, cols = linear_sum_assignment(np.where(cost <= self.gate, cost, BIG))
        pairs = [(tracks[r], free[c]) for r, c in zip(rows, cols) if cost[r, c] <= self.gate]
        matched = {tr.id for tr, _ in pairs}
        return pairs, [tr for tr in tracks if tr.id not in matched]

    def step(self, t, Z):
        """
        
        """
        dt = 0.0 if self.t is None else t - self.t
        self.t = t
        F, Q = cv_matrices(dt, self.accel_std)
        for tr in self.tracks:
            tr.kf.predict(F, Q)

        free = list(range(len(Z)))
        used = []
        trusted = [tr for tr in self.tracks if self.confirmed(tr)]
        probation = [tr for tr in self.tracks if not self.confirmed(tr)]
        for group in (trusted, probation):
            pairs, unmatched = self._match(group, free, Z)
            for tr, j in pairs:
                tr.kf.update_nonlinear(
                    Z[j],
                    partial(radar_measurement, radar_pos=self.radar_pos),
                    partial(radar_jacobian, radar_pos=self.radar_pos),
                    angle_rows=(1,),
                )
                tr.hits += 1
                tr.misses = 0
                free.remove(j)
                used.append((tr.id, j))
            for tr in unmatched:
                tr.misses += 1

        # drop tracks that stopped getting dots (probation tracks get no second chance)
        self.tracks = [tr for tr in self.tracks
                       if tr.misses <= (self.max_misses if self.confirmed(tr) else 0)]

        # any dot nobody claimed starts a new track
        for j in free:
            kf = radar_tracker(1.0, self.accel_std, self.range_std, self.bearing_std,
                               Z[j], self.radar_pos)
            kf.P = np.diag([200.0**2, 200.0**2, self.vel_std**2, self.vel_std**2])
            self.tracks.append(Track(self.next_id, kf, t))
            used.append((self.next_id, j))
            self.next_id += 1

        for tr in self.tracks:
            tr.path.append((t, tr.kf.x[0], tr.kf.x[1]))
        return used