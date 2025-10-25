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


def pos_to_tpd(pos):
    x, y, z = pos
    theta = math.atan2(y, x)
    phi = math.atan2(z, math.sqrt(x ** 2 + y ** 2))
    distance = math.sqrt(x ** 2 + y ** 2 + z ** 2)

    return np.array([theta, phi, distance])


def pos_to_tpd_jacob(pos):
    x, y, z = pos

    r_sq = x ** 2 + y ** 2
    r = math.sqrt(r_sq)

    d_sq = r_sq + z ** 2
    d = math.sqrt(d_sq)

    if r < 1e-6:
        return np.eye(3)

    return np.array([
        [-y / r_sq, x / r_sq, 0],
        [-x * z / (d_sq * r), -y * z / (d_sq * r), r / d_sq],
        [x / d, y / d, z / d]
    ])


def world_to_main_screen(world_pos, screen_center, world_scale):
    screen_x = screen_center[0] + world_pos[0] * world_scale
    screen_y = screen_center[1] + world_pos[1] * world_scale

    return np.array([int(screen_x), int(screen_y)])


def camera_to_screen(camera_screen_pos):
    if camera_screen_pos <= 0:
        return None

    # 透视投影
    x_norm = camera_screen_pos[1] / camera_screen_pos[0]  # y / x
    y_norm = camera_screen_pos[2] / camera_screen_pos[0]  # z / x


def limit_rad(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi



