import math
import numpy as np


def limit_rad(angle: float) -> float:
    """限制角度在[-pi, pi]范围内"""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


# 辅助函数
def safe_angle_sub(angle1: float, angle2: float) -> float:
    """安全的角度减法"""
    angle = angle1 - angle2
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def get_distance(pose1, pose2):
    """计算3维位姿之间的距离"""
    dx = pose1[0] - pose2[0]
    dy = pose1[1] - pose2[1]
    dz = pose1[2] - pose2[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)

