import math
import numpy as np


def xyz2ypd(xyz):
    """3D坐标转角度距离 - 遵循源码逻辑"""
    x, y, z = xyz
    yaw = math.atan2(y, x)
    distance = math.sqrt(x ** 2 + y ** 2 + z ** 2)
    pitch = math.atan2(z, math.sqrt(x ** 2 + y ** 2))
    return np.array([yaw, pitch, distance])


def xyz2ypd_jacobian(xyz):
    """坐标到角度距离的雅可比矩阵"""
    x, y, z = xyz
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


def limit_rad(angle):
    """限制角度在[-π, π]范围内"""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle <= -math.pi:
        angle += 2 * math.pi
    return angle

