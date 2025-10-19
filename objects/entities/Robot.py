import numpy as np
import yaml
from enum import Enum

from object.entity.Armor import Armor


# 车的型号，顺便排列优先级
class RobotType(Enum):
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

        self.robot_type = robot_type
        self.priority = robot_type  # 车的型号决定了打击的优先级
        self.armor_count = 0
        self.armor_size = ''

        self.length = 0
        self.width = 0
        self.high_height = 0
        self.low_height = 0

        self.world_pos = np.array([0., 0., 0.])
        self.world_vel = np.array([0., 0., 0.])
        self.world_rpy = np.array([0., 0., 0.])
        self.world_omg = np.array([0., 0., 0.])

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

        for i in range(self.armor_count):  # 车的规格设定为长低短高，按 装甲板半径从长到短的顺序，对装甲板进行逆时针编号
            armor = Armor(i)

            armor.armor_size = self.armor_size

            if self.armor_count == 4:
                if i % 2 == 0:
                    armor.world_pos[0] = self.world_pos[0] - self.length / 2
                    armor.world_pos[1] = self.world_pos[1]
                    armor.world_pos[2] = self.low_height
                    armor.radius = self.length / 2
                else:
                    armor.world_pos[0] = self.world_pos[0]
                    armor.world_pos[1] = self.world_pos[1] + self.width / 2
                    armor.world_pos[2] = self.high_height
                    armor.radius = self.width / 2
            elif self.armor_count == 2:  # Sentry是双装甲板，设置为low height，id为 0，1
                armor.world_pos[0] = self.world_pos[0] - self.length / 2
                armor.world_pos[1] = self.world_pos[1]
                armor.world_pos[2] = self.low_height
                armor.radius = self.length / 2

            self.armors.append(armor)





# hero = Robot(RobotType.Sentry)
# hero.initialization()
# armor = hero.armors[1]
# print(armor.armor_size, armor.armor_id, armor.pos)



