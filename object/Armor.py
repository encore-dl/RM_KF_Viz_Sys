from enum import Enum
import numpy as np


# 装甲板类型和名称枚举
class ArmorType(Enum):
    SMALL = 0
    BIG = 1


class ArmorName(Enum):
    UNKNOWN = 0
    OUTPOST = 1
    HERO = 2
    ENGINEER = 3
    INFANTRY3 = 4
    INFANTRY4 = 5
    INFANTRY5 = 6
    SENTRY = 7


class Armor:
    """装甲板类 - 严格遵循源码结构"""

    def __init__(self):
        self.name = ArmorName.UNKNOWN
        self.type = ArmorType.SMALL
        self.priority = 1
        self.color = 0  # 0 for blue, 1 for red

        # 3D坐标
        self.xyz_in_world = np.zeros(3)
        self.xyz_in_gimbal = np.zeros(3)

        # 姿态角
        self.ypr_in_world = np.zeros(3)  # yaw, pitch, roll
        self.ypr_in_gimbal = np.zeros(3)
        self.ypd_in_world = np.zeros(3)  # yaw, pitch, distance

        # 图像坐标
        self.center = np.zeros(2)
        self.center_norm = np.zeros(2)
        self.points = []  # 4个角点

        # 优化用
        self.yaw_raw = 0.0