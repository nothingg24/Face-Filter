import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag

class KalmanFilter(KalmanFilter):
    def __init__(self) -> None:
        super().__init__(dim_x=4, dim_z=2)
        dt = 1.0
        self.F = np.array([[1, dt, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, dt],
                           [0, 0, 0, 1]])
        self.alpha = 1.02
        self.Q = block_diag(Q_discrete_white_noise(dim=2, dt=dt, var=0.001))

