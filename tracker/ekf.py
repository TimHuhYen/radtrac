import numpy as np

from tracker.kalman import KalmanFilter

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
        z = np.asarray(z, gtype=float)
        R = self.R if R is None else np.asarray(R, dtype=float)

        H = jacobian(self.x)
        y = z - h(self.x)
        for i in angle_rows:
            y[i] = (y[i] + np.pi) % (2 * np.pi) - np.pi

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(len(self.x))
        self.P = (I - K @ H) @ self.P