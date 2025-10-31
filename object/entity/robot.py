import numpy as np
import math
import yaml
from enum import IntEnum

from object.entity.armor import Armor
from utils.math_tool import pos_to_tpd


# 车的型号，顺便排列优先级
class RobotType(IntEnum):
    Hero = 1
    Sentry = 2
    Infantry = 3
    Engineer = 4
    Outpost = 5
    Base = 6

    @classmethod
    def get_name(cls, robot_type):
        robot_name_str = ''
        if robot_type == RobotType.Hero:
            robot_name_str = 'Hero'
        elif robot_type == RobotType.Sentry:
            robot_name_str = 'Sentry'
        elif robot_type == RobotType.Infantry:
            robot_name_str = 'Infantry'
        elif robot_type == RobotType.Engineer:
            robot_name_str = 'Engineer'
        return robot_name_str


class Robot:
    def __init__(self, robot_type):
        self.armors = []

        self.world_pos = np.array([0., 0., 0.])
        self.world_vel = np.array([0., 0., 0.])
        self.world_tpd = np.array([0., 0., 0.])
        self.world_rpy = np.array([0., 0., 0.])
        self.world_omg = np.array([0., 0., 0.])

        self.robot_type = robot_type
        self.priority = robot_type  # 车的型号决定了打击的优先级
        self.armor_count = 0
        self.armor_size = ''

        self.length = 0
        self.width = 0
        self.high_height = 0
        self.low_height = 0
        self.radius = 0

        self.load_config()

    def load_config(self):
        with open('data/config.yaml', 'r') as file:
            data = yaml.safe_load(file)

        robot_name_str = RobotType.get_name(self.robot_type)

        self.armor_count = data['Robot'][robot_name_str]['armor_count']
        self.armor_size = data['Robot'][robot_name_str]['armor_size']
        self.length = data['Robot'][robot_name_str]['length']
        self.width = data['Robot'][robot_name_str]['width']
        self.high_height = data['Robot'][robot_name_str]['high_height']
        self.low_height = data['Robot'][robot_name_str]['low_height']
        self.radius = self.length / 2

        for i in range(self.armor_count):  # 车的规格设定为长低短高，按 装甲板半径从长到短的顺序，对装甲板进行逆时针编号，最x正的开始
            armor = Armor(i)

            armor.armor_size = self.armor_size
            armor.robot_type = self.robot_type
            armor.priority = self.priority

            # 世界坐标系：x 前 y 右 z 上
            # 逆时针为：y正 x正 y负 x负
            # 方向角来看，是 x正时yaw = 0，顺时针为正
            if self.armor_count == 4:
                if i % 2 == 0:  # 前后装甲板
                    if i == 0:  # 前装甲
                        armor.world_pos[0] = self.world_pos[0] + self.length / 2
                        armor.world_rpy[2] = 0.
                    elif i == 2:  # 后装甲
                        armor.world_pos[0] = self.world_pos[0] - self.length / 2
                        armor.world_rpy[2] = math.pi
                    armor.world_pos[1] = self.world_pos[1]
                    armor.world_pos[2] = self.low_height
                    armor.radius = self.length / 2
                else:  # 左右装甲板
                    armor.world_pos[0] = self.world_pos[0]
                    if i == 1:  # 右装甲
                        armor.world_pos[1] = self.world_pos[1] + self.width / 2
                        armor.world_rpy[2] = math.pi / 2
                    elif i == 3:  # 左装甲
                        armor.world_pos[1] = self.world_pos[1] - self.width / 2
                        armor.world_rpy[2] = -math.pi / 2
                    armor.world_pos[2] = self.high_height
                    armor.radius = self.width / 2
            elif self.armor_count == 2:  # Sentry是双装甲板，设置为low height，id为 0，1
                if i == 0:
                    armor.world_pos[0] = self.world_pos[0] + self.length / 2
                    armor.world_rpy[2] = 0.
                elif i == 1:
                    armor.world_pos[0] = self.world_pos[0] - self.length / 2
                    armor.world_rpy[2] = math.pi
                armor.world_pos[1] = self.world_pos[1]
                armor.world_pos[2] = self.low_height
                armor.radius = self.length / 2

            armor.world_tpd = pos_to_tpd(armor.world_pos)

            self.armors.append(armor)

        self.world_pos[2] = np.mean([armor.world_pos[2] for armor in self.armors])
        self.world_tpd = pos_to_tpd(self.world_pos)



