import numpy as np
import math

from core.entities.rigid.rigid import Rigid
from core.entities.rigid.armor import Armor
from config.config_manager import cfg_mgr
from core.algorithms.math import euler_to_rotation_matrix, pos_to_tpd, limit_rad


class Robot(Rigid):
    def __init__(self, robot_type, **kwargs):
        super().__init__(**kwargs)
        cfg = cfg_mgr.get_robot_config(robot_type)

        self.armors = []

        self.robot_type = robot_type
        self.priority = robot_type  # 车的型号决定了打击的优先级
        self.armor_count = cfg.armor_count
        self.armor_size = cfg.armor_size

        self.length = cfg.length
        self.width = cfg.width
        self.high_height = cfg.high_height
        self.low_height = cfg.low_height
        self.radius = self.length / 2
        self.light_bar_interval = cfg.light_bar_interval
        self.light_bar_length = cfg.light_bar_length

        if self.armor_count == 4:
            center_z = (self.high_height + self.low_height) / 2.0
        else:
            center_z = self.low_height
        self.world_pos[2] = center_z
        self.world_tpd = pos_to_tpd(self.world_pos)

        self._create_armors()
        self.update_armors()

    def _create_armors(self):
        center_z = self.world_pos[2]
        for i in range(self.armor_count):
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
                    rel_rpy = np.array([0, 0, -math.pi / 2])
                elif i == 2:  # 后
                    rel_pos = np.array([-self.length / 2, 0, offset_z_low])
                    rel_rpy = np.array([0, 0, math.pi])
                elif i == 3:  # 左
                    rel_pos = np.array([0, -self.width / 2, offset_z_high])
                    rel_rpy = np.array([0, 0, math.pi / 2])
            elif self.armor_count == 2:
                offset_z = self.low_height - center_z
                if i == 0:  # 前
                    rel_pos = np.array([self.length / 2, 0, offset_z])
                    rel_rpy = np.array([0, 0, 0])
                elif i == 1:  # 后
                    rel_pos = np.array([-self.length / 2, 0, offset_z])
                    rel_rpy = np.array([0, 0, math.pi])

            armor.mount_pos = rel_pos
            armor.mount_R = euler_to_rotation_matrix(rel_rpy)

            self.armors.append(armor)

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





