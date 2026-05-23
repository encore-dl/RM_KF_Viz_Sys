import numpy as np
import math
from core.entities.rigid.rigid import Rigid
from core.entities.rigid.armor import Armor
from config.config_manager import cfg_mgr
from core.algorithms.math import euler_to_rotation_matrix, pos_to_tpd
from core.entities.property.robot_type import RobotType


class Outpost(Rigid):
    """前哨站（塔）实体，固定位置，恒速旋转，三个装甲板"""
    def __init__(self, world_pos=None, **kwargs):
        cfg = cfg_mgr.get_outpost_config()
        default_pos = np.array([0.0, 0.0, cfg.base_height + cfg.height_step])
        if world_pos is None:
            world_pos = default_pos.copy()
        super().__init__(world_pos=world_pos, **kwargs)

        self.radius = cfg.radius
        self.armor_count = cfg.armor_count
        self.base_height = cfg.base_height
        self.height_step = cfg.height_step
        self.rotate_speed = cfg.rotate_speed

        # 初始化装甲板
        self.armors = []
        self._create_armors()
        self.update_armors()

    def _create_armors(self):
        center_z = self.world_pos[2]  # 中心高度 = 中间装甲板高度
        for i in range(self.armor_count):
            armor = Armor(armor_id=i, robot_type=RobotType.Outpost)

            if i == 0:
                armor_height = self.base_height
                angle = 0.0
            elif i == 1:
                armor_height = self.base_height + self.height_step
                angle = 2.0 * math.pi / 3.0   # 120°
            else:  # i == 2
                armor_height = self.base_height + 2.0 * self.height_step
                angle = 4.0 * math.pi / 3.0   # 240°

            # 相对中心的位置
            rel_x = self.radius * math.cos(angle)
            rel_y = self.radius * math.sin(angle)
            rel_z = armor_height - center_z
            rel_pos = np.array([rel_x, rel_y, rel_z])

            # 装甲板自身朝向：径向向外
            rel_rpy = np.array([0.0, 0.0, angle])
            armor.mount_pos = rel_pos
            armor.mount_R = euler_to_rotation_matrix(rel_rpy)

            self.armors.append(armor)

    def update_armors(self):
        R = euler_to_rotation_matrix(self.world_rpy)
        for i, armor in enumerate(self.armors):
            armor.world_pos = self.world_pos + R @ armor.mount_pos
            armor_world_R = R @ armor.mount_R
            armor.world_rpy = self.world_rpy.copy()
            armor.world_rpy[2] += i * 2 * math.pi / self.armor_count

            r_vec = armor.world_pos - self.world_pos
            armor.world_vel = self.world_vel + np.cross(self.world_omg, r_vec)
            armor.world_omg = self.world_omg.copy()
            armor.world_tpd = pos_to_tpd(armor.world_pos)

            for j, local_pt in enumerate(armor.init_light_corners):
                armor.light_corners[j] = armor.world_pos + armor_world_R @ local_pt

    def update_outpost(self, dt):
        # 恒速旋转
        self.world_omg[2] = self.rotate_speed
        self.world_rpy[2] += self.world_omg[2] * dt
        self.world_rpy[2] = (self.world_rpy[2] + math.pi) % (2 * math.pi) - math.pi
        self.update_armors()