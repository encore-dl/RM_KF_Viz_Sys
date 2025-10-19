import math
import numpy as np


def get_euler_rotate_matrix(rpy):
    R_x = np.array([
        [1., 0., 0.],
        [0., math.cos(rpy[0]), -math.sin(rpy[0])],
        [0., math.sin(rpy[0]), math.cos(rpy[0])]
    ])

    R_y = np.array([
        [math.cos(rpy[1]), 0., math.sin(rpy[1])],
        [0., 1., 0.],
        [-math.sin(rpy[1]), 0., math.cos(rpy[1])]
    ])

    R_z = np.array([
        [math.cos(rpy[2]), -math.sin(rpy[2]), 0.],
        [math.sin(rpy[2]), math.cos(rpy[2]), 0.],
        [0., 0., 1.]
    ])

    R = R_z @ R_y @ R_x

    return R


