import numpy as np
import copy

from core.entities.rigid.rigid import Rigid
from config.config_manager import cfg_mgr


class Armor(Rigid):
    def __init__(self, armor_id, robot_type, **kwargs):
        super().__init__(**kwargs)
        cfg = cfg_mgr.get_robot_config(robot_type)

        self.armor_id = armor_id  # 装甲板会有多个，每个装甲板先拥有自己的id
        self.robot_type = robot_type
        self.priority = robot_type

        self.armor_size = cfg.armor_size
        self.light_bar_interval = cfg.light_bar_interval
        self.light_bar_length = cfg.light_bar_length

        w = self.light_bar_interval
        h = self.light_bar_length
        self.init_light_corners = [
            # 假设 Y+ 是左 (Left)，Z+ 是上 (Up) -> 符合 FLU 坐标系下的左侧
            # 顺序: 左上 -> 右上 -> 右下 -> 左下
            np.array([0,  w/2.,  h/2.]),  # 左上
            np.array([0, -w/2.,  h/2.]),  # 右上
            np.array([0, -w/2., -h/2.]),  # 右下
            np.array([0,  w/2., -h/2.]),  # 左下
        ]
        self.light_corners = copy.deepcopy(self.init_light_corners)

        self.mount_pos = np.zeros(3)
        self.mount_R = np.eye(3)




