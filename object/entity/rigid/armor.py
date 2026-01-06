import numpy as np
import yaml
import copy

from object.entity.rigid.rigid import Rigid
from object.entity.rigid.robot import RobotType


class Armor(Rigid):
    def __init__(self, armor_id, robot_type, **kwargs):
        super().__init__(**kwargs)

        self.armor_id = armor_id  # 装甲板会有多个，每个装甲板先拥有自己的id
        self.robot_type = robot_type
        self.priority = robot_type

        self.armor_size = ''
        self.radius = 0  # 绕车体中心旋转半径
        self.light_bar_interval = 0.
        self.light_bar_length = 0.

        self.init_light_bar_endpoints = []
        self.light_bar_endpoints = []

        self.mount_pos = np.zeros(3)
        self.mount_R = np.eye(3)

        self._load_config()

    def _load_config(self):
        with open('../data/config.yaml', 'r') as file:
            data = yaml.safe_load(file)

        robot_name_str = RobotType.get_name(self.robot_type)

        self.armor_size = data['Robot'][robot_name_str]['armor_size']
        self.light_bar_interval = data['Robot'][robot_name_str]['light_bar_interval']
        self.light_bar_length = data['Robot'][robot_name_str]['light_bar_length']

        self.init_light_bar_endpoints = [
            np.array([0, -(self.light_bar_interval/2.), (self.light_bar_length/2.)]),
            np.array([0, -(self.light_bar_interval/2.), -(self.light_bar_length/2.)]),
            np.array([0, (self.light_bar_interval/2.), -(self.light_bar_length/2.)]),
            np.array([0, (self.light_bar_interval/2.), (self.light_bar_length/2.)]),
        ]
        self.light_bar_endpoints = copy.deepcopy(self.init_light_bar_endpoints)






