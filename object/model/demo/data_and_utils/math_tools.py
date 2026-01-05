import numpy as np


def limit_rad(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def safe_angle_sub(angle1, angle2):
    return limit_rad(angle1 - angle2)


def unwrap_angle(angle, ref_angle):
    diff = safe_angle_sub(angle, ref_angle)
    return ref_angle + diff



