from functools import partial

import numpy as np
from scipy.optimize import linear_sum_assignment

from tracker.ekf import radar_tracker
from tracker.kalman import cv_matrices
from tracker.sensors import radar_jacobian, radar_measurement

BIG = 1e6 # cost for a pairing that fails the gate