import numpy as np

class KalmanFilter:
    """
    Simple Kalman Filter implementation.
    x =  Initial state estimate
    P =  Initial estimate covariance
    F =  State transition matrix
    H =  Observation matrix
    Q =  Process noise covariance
    R =  Measurement noise covariance

    x is the state estimate (flat arr)
    P is its uncertainty
    F moves the state forward in time
    H maps the state to what the sensor measures
    Q is the doubt added each stop
    R is the sensor noise
    """

    def __init__(self, F, H, Q, R, x0, P0):
        self.F = np.asarray(F, dtype=float)
        self.H = np.asarray(H, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.x = np.asarray(x0, dtype=float)
        self.P = np.asarray(P0, dtype=float)

    def predict(self):
        # Project estimate forward
        self.x = self.F @ self.x
        # Add doubt for time passed
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        pass
