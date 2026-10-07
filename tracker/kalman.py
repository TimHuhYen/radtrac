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
        z = np.asarray(z, dtype=float)
        y = z - self.H @ self.x                     # suprise
        S = self.H @ self.P @ self.H.T + self.R     # expected suprise
        K = self.P @ self.H.T @ np.linalg.inv(S)    # trust weight gain
        self.x = self.x + K @ y                     # blend reading
        I = np.eye(len(self.x))
        self.P = (I - K @ self.H) @ self.P          # shrink uncer

    def constant_velocity_2d(dt, accel_std, meas_std, x0):
        """
        Build a filter for a plane that mostly keeps its current velocity.

        State is [x position, y position, x velocity, y velocity].
        dt: seconds between readings.
        accel_std: how much unexpected acceleration to allow (m/s&2).
        meas_std: sensor position errot in meters.
        z0: first reading [x, y], used to start the filter.
        """
        F = np.array([ 
            [1, 0 , dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Observation matrix
        # ***position
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        """
        Random acceleration doubt, for one axis (pos, velocity)
        Q adds uncertainty to P at every prediction step
            accel_std is how much surprise acceleration we expect
        The filter will assume that the object can accelerate at this rate,
            and will increase the uncertainty of the position and velocity accordingly.
        """
        q = accel_std ** 2
        block = q * np.array([
            [dt**4 / 4, dt**3 / 2],
            [dt**3 / 2, dt**2],
        ])

        Q = np.zeros((4, 4))
        Q[np.ix_([0, 2], [0, 2])] = block # x position and x velocity
        Q[np.ix_([1, 3], [1, 3])] = block # y position and y velocity

        """
        Sensorys error covariance matrix, 
            H = for one axis (pos, velocity)
        """
        R = np.eye(2) * meas_std ** 2

        x0 = [z0[0], z0[1], 0, 0]
        P0 = np.diag([meas_std**2, meas_std**2, 300.0**2, 200.0**2])
        return KalmanFilter(F, H, Q, R, x0, P0)
    