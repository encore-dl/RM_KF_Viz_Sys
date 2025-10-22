import numpy as np


class Armor:
    def __init__(self, armor_id=0):
        self.armor_id = armor_id  # 装甲板会有多个，每个装甲板先拥有自己的id
        self.armor_size = ''
        self.robot_type = None
        self.priority = None

        self.world_pos = np.array([0., 0., 0.])
        self.world_vel = np.array([0., 0., 0.])
        self.world_rpy = np.array([0., 0., 0.])
        self.radius = 0


