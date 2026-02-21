import math

import numpy as np


def euler_to_rotation_matrix(rpy):
    roll, pitch, yaw = rpy

    # 绕 X 轴旋转 (Roll)
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
    ])

    # 绕 Y 轴旋转 (Pitch)
    Ry = np.array([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])

    # 绕 Z 轴旋转 (Yaw)
    Rz = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ])

    # 旋转顺序：先转X(Roll)，再转Y(Pitch)，最后转Z(Yaw)
    # 内旋还是外旋？
    R = Rz @ Ry @ Rx
    return R


def rotation_matrix_to_euler(R):
    sqrt_p = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sqrt_p < 1e-6

    if not singular:
        r = math.atan2(R[2, 1], R[2, 2])
        p = math.atan2(-R[2, 0], sqrt_p)
        y = math.atan2(R[1, 0], R[0, 0])
    else:
        r = math.atan2(-R[1, 2], R[1, 1])
        p = math.atan2(-R[2, 0], sqrt_p)
        y = 0
    return np.array([r, p, y])


def get_rigid_transform(pos, rpy):
    R = euler_to_rotation_matrix(rpy)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


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


def world_to_main_screen(world_pos, main_screen_center, world_scale):
    """
    将世界坐标 (X前, Y左) 映射到 2D 俯视地图 (u右, v下)
    Map View:
        Screen Up (v-)    <-- World X+ (Forward)
        Screen Right (u+) <-- World Y- (Right)
    """
    # 屏幕 x (u) 对应 世界 -y (右)
    screen_x = main_screen_center[0] - world_pos[1] * world_scale
    # 屏幕 y (v) 对应 世界 -x (后) -> 因为屏幕上方是y=0，世界前方是x+
    # 所以 世界x+ 应该是 屏幕上方
    screen_y = main_screen_center[1] - world_pos[0] * world_scale

    return np.array([int(screen_x), int(screen_y)])


def world_to_camera_screen(world_pos, camera, camera_screen_center, resolution):
    pixel_pos = camera.world_to_pixel(world_pos)
    if pixel_pos is not None:
        u, v = pixel_pos

        if -int(resolution[0]/2) <= u < int(resolution[0]/2) and -int(resolution[1]/2) <= v < int(resolution[1]/2):
            u += camera_screen_center[0]
            v += camera_screen_center[1]

            return u, v
    else:
        return None
