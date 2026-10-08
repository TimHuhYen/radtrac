import numpy as np

from tracker.sensors import radar_measurement

def simulate_track(n_steps=300, dt=1.0, speed=220.0, meas_std=50.0, seed=0):
    """
    Fly a plane wit two gentle turns, then add sensor noise.
    
    Returns (truth, readings). truth has columns [x, y, x speed, y speed];
    readings has columns [x, y] with Gaussion noise of meas_std meters.
    """

    rng = np.random.default_rng(seed)

    # turn rate in radians per second: straight, left, straight, right straight
    turn_rate = np.zeros(n_steps)
    turn_rate[80:130] = np.deg2rad(2.0)
    turn_rate[180:230] = np.deg2rad(-2.5)

    heading = 0.3
    pos = np.array([0.0, 0.0])
    truth = np.zeros((n_steps, 4))
    for k in range(n_steps):
        heading += turn_rate[k] * dt
        vel = speed * np.array([np.cos(heading), np.sin(heading)])
        pos = pos + vel * dt
        truth[k] = [pos[0], pos[1], vel[0], vel[1]]

    readings = truth[:, :2] + rng.normal(0.0, meas_std, size=(n_steps, 2))
    return truth, readings

def rmse(estimate_xy, truth_xy):
    """
    Root mean square error: the typical miss distance in meters.
    """
    diff = np.asarray(estimate_xy) - np.asarray(truth_xy)
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))

def simulate_radar_readings(truth, radar_pos, range_std=50.0, bearing_std=0.002, seed=1):
    """
    What a radar at radar_pos report for each true pos
    
    Each reading is [range(m), bearing (radians)] with random errors added.
    """
    rng = np.random.default_rng(seed)
    clean = np.array([radar_measurement(state, radar_pos) for state in truth])
    noise = rng.nromal(0.0, [range_std, bearing_std], size=clean.shape)
    return clean + noise