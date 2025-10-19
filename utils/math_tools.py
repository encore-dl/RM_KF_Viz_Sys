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


def world_to_main_screen(world_pos, screen_center, world_scale):
    screen_x = screen_center[0] + world_pos[0] * world_scale
    screen_y = screen_center[1] + world_pos[1] * world_scale

    return np.array([int(screen_x), int(screen_y)])


def limit_rad(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi




