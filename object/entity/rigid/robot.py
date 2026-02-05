import numpy as np
import math
import yaml

from object.entity.rigid.rigid import Rigid
from object.entity.property.robot_type import RobotType
from object.entity.rigid.armor import Armor
from utils.math_tool import (
    pos_to_tpd,
    limit_rad,
    euler_to_rotation_matrix,
    rotation_matrix_to_euler
)


class Robot(Rigid):
    def __init__(self, robot_type, **kwargs):
        super().__init__(**kwargs)

        self.armors = []

        self.robot_type = robot_type
        self.priority = robot_type  # 车的型号决定了打击的优先级
        self.armor_count = 0
        self.armor_size = ''

        self.length = 0.
        self.width = 0.
        self.high_height = 0.
        self.low_height = 0.
        self.radius = 0.
        self.light_bar_interval = 0.
        self.light_bar_length = 0.

        self.load_config()

    def load_config(self):
        with open('../data/config.yaml', 'r') as file:
            data = yaml.safe_load(file)

        robot_name_str = RobotType.get_name(self.robot_type)

        self.armor_count = data['Robot'][robot_name_str]['armor_count']
        self.armor_size = data['Robot'][robot_name_str]['armor_size']
        self.length = data['Robot'][robot_name_str]['length']
        self.width = data['Robot'][robot_name_str]['width']
        self.high_height = data['Robot'][robot_name_str]['high_height']
        self.low_height = data['Robot'][robot_name_str]['low_height']
        self.radius = self.length / 2

        if self.armor_count == 4:
            center_z = (self.high_height + self.low_height) / 2.
        else:
            center_z = self.low_height

        self.world_pos[2] = center_z

        for i in range(self.armor_count):  # 车的规格设定为长低短高，按 装甲板半径从长到短的顺序，对装甲板进行逆时针编号，最x正的开始
            armor = Armor(
                armor_id=i,
                robot_type=self.robot_type
            )

            rel_pos = np.zeros(3)
            rel_rpy = np.zeros(3)

            offset_z_high = self.high_height - center_z
            offset_z_low = self.low_height - center_z

            if self.armor_count == 4:
                # 四装甲板布局
                if i == 0:  # 前
                    rel_pos = np.array([self.length / 2, 0, offset_z_low])
                    rel_rpy = np.array([0, 0, 0])
                elif i == 1:  # 右
                    rel_pos = np.array([0, self.width / 2, offset_z_high])
                    rel_rpy = np.array([0, 0, -math.pi / 2])  # 假设右侧装甲板朝向右(-90度)
                elif i == 2:  # 后
                    rel_pos = np.array([-self.length / 2, 0, offset_z_low])
                    rel_rpy = np.array([0, 0, math.pi])
                elif i == 3:  # 左
                    rel_pos = np.array([0, -self.width / 2, offset_z_high])
                    rel_rpy = np.array([0, 0, math.pi / 2])

            elif self.armor_count == 2:
                offset_z = self.low_height - center_z

                # 哨兵双板布局 (假设前后)
                if i == 0:  # 前
                    rel_pos = np.array([self.length / 2, 0, offset_z])
                    rel_rpy = np.array([0, 0, 0])
                elif i == 1:  # 后
                    rel_pos = np.array([-self.length / 2, 0, offset_z])
                    rel_rpy = np.array([0, 0, math.pi])

            # === 第二步：将安装参数存入 Armor 对象 ===
            # 这些参数一旦设定，通常不再改变
            armor.mount_pos = rel_pos
            armor.mount_R = euler_to_rotation_matrix(rel_rpy)

            self.armors.append(armor)

        self.world_tpd = pos_to_tpd(self.world_pos)

        self.update_armors()

    def update_armors(self):
        R_robot = euler_to_rotation_matrix(self.world_rpy)

        for armor in self.armors:
            armor.world_pos = self.world_pos + (R_robot @ armor.mount_pos)
            armor_world_R = R_robot @ armor.mount_R

            armor.world_rpy = self.world_rpy.copy()
            # 加上装甲板自身的安装角度 (假设均匀分布)
            angle_offset = (armor.armor_id * 2 * math.pi / self.armor_count)
            armor.world_rpy[2] = limit_rad(self.world_rpy[2] + angle_offset)
            # armor.world_rpy[2] = limit_rad(robot.world_rpy[2] + math.pi + angle_offset)

            r_vec = armor.world_pos - self.world_pos
            armor.world_vel = self.world_vel + np.cross(self.world_omg, r_vec)

            armor.world_omg = self.world_omg.copy()

            armor.world_tpd = pos_to_tpd(armor.world_pos)

            for i, local_point in enumerate(armor.init_light_corners):
                armor.light_corners[i] = armor.world_pos + (armor_world_R @ local_point)





