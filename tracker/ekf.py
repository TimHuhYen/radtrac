import numpy as np

from tracker.kalman import KalmanFilter, constant_velocity_2d
from tracker.sensors import radar_to_xy

class ExtendedKalmanFilter(KalmanFilter):
    """
    Kalman filter, sensor is curved function of the state.
    
    predict() is inherited unchanged, cause the planes motion model
    is a straight matrix mult. Only the update step changes.
    """

    def update_nonlinear(self, z, h, jacobian, R=None, angle_rows=()):
        """
        Blend in reading z from a sensor that follows a curved formula

        h(x): rela formula, giving reading the sensor should use
        jacobian(x): the matrix of slops of h at x, used in place of big H !!!
        R: sensor noise from reading (defaults to filter's R)
        angle_rows: which entries of z are angles that wrap around
        """
        z = np.asarray(z, dtype=float)
        
        R = self.R if R is None else np.asarray(R, dtype=float)
        H = jacobian(self.x)                                # curr estimate slope
        y = z - h(self.x)                                   # real curve surpise
        for i in angle_rows:
            y[i] = (y[i] + np.pi) % (2 * np.pi) - np.pi     # wrap into [-pi, pi]

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(len(self.x))
        self.P = (I - K @ H) @ self.P

def radar_tracker(dt, accel_std, range_std, bearing_std, first_reading, radar_pos, pos_std=200.0):
    """
    Building external filter that starts from a first radar reading.
    """
    base = constant_velocity_2d(dt, accel_std, 1.0, [0.0, 0.0])
    R = np.diag([range_std**2, bearing_std**2])

    xy = radar_to_xy(first_reading, radar_pos)
    x0 = [xy[0], xy[1], 0.0, 0.0]
    P0 = np.diag([pos_std**2, pos_std**2, 300.0**2, 300.0**2])
    return ExtendedKalmanFilter(base.F, base.H, base.Q, R, x0, P0)