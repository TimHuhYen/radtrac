import numpy as np

# shrimple measurement functions for diff sensirs

def radar_measurement(state, radar_pos=(0.0, 0.0)):
    """
    Ideal radar at radar_pos reports: [range (m), bearing (radians)]

    state: [x, y, vx, vy]
    radar_pos: (x, y) of the radar
    """
    dx = state[0] - radar_pos[0]
    dy = state[1] - radar_pos[1]
    return np.array([np.hypot(dx, dy), np.arctan2(dy, dx)])

def radar_jacobian(state, radar_pos=(0.0, 0.0)):
    """
    """
    dx = state[0] - radar_pos[0]
    dy = state[1] - radar_pos[1]
    r2 = dx**2 + dy**2
    r = np.sqrt(r2)
    return np.array([
        [dx / r, dy / r, 0.0, 0.0],
        [-dy / r2, dx / r2, 0.0, 0.0]
    ])