import numpy as np


class Armor:
    def __init__(self, armor_id=0):
        self.armor_id = armor_id  # 装甲板会有多个，每个装甲板先拥有自己的id
        self.armor_size = ''
        self.robot_type = None
        self.priority = None

        self.world_pos = np.array([0., 0., 0.])  # 世界坐标
        self.world_vel = np.array([0., 0., 0.])  # 世界速度
        self.world_tpd = np.array([0., 0., 0.])  # 世界球坐标
        self.world_rpy = np.array([0., 0., 0.])  # 世界朝向角
        self.world_omg = np.array([0., 0., 0.])  # 世界角速度
        self.radius = 0  # 绕车体中心旋转半径


