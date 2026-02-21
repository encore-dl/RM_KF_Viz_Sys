import math


def limit_rad(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def safe_angle_sub(angle1, angle2):
    return limit_rad(angle1 - angle2)
