import math
import time

import numpy as np

from object.model.tjurm.data_and_utils.config import TJURMConfig
from object.model.tjurm.submodel.track_queue import TrackQueue
from object.model.tjurm.submodel.antitop import Antitop


class TJURMModel:
    """TJURM主模型，整合TrackQueue和Antitop"""

    def __init__(self):
        self.config = TJURMConfig()

        self.track_queue = TrackQueue(
            self.config.track_count,
            self.config.track_distance,
            self.config.track_delay,
            self.config.track_fire_interval,
            self.config.track_fire_high_delay,
            self.config.slide_integrator
        )
        self.antitop = Antitop(
            self.config.antitop_min_r,
            self.config.antitop_max_r,
            4,
            False,
            self.config.antitop_fire_retention,
            self.config.track_fire_interval,
            self.config.track_fire_high_delay,
            0.002,
            self.config.slide_integrator
        )

        self.flag_antitop = False
        self.flag_center = False
        self.flag_str = ""

        self.last_armor_id = None
        self.armor_count = 0
        self.update_count = 0

        self.rotate_delay = self.config.rotate_delay

    def push(self, obsrv_armor, t):
        pose = np.array([
            obsrv_armor.world_pos[0],
            obsrv_armor.world_pos[1],
            obsrv_armor.world_pos[2],
            obsrv_armor.world_rpy[2]
        ])
        t = t

        self.track_queue.push(pose, t)

    def update(self):
        self.track_queue.update()

        is_alright, pose, t = self.track_queue.get_antitop_input()

        if not is_alright:
            return

        self.antitop.push(pose, t)

    def get_pred_pos(self):
        fly_delay = 0.
        pred_pos = []

        # 角速度判断
        omega = self.antitop.get_omega()
        if abs(omega) > self.config.track_to_antitop:
            self.flag_antitop = True
        elif abs(omega) < self.config.antitop_to_track:
            self.flag_antitop = False

        if abs(omega) > self.config.armor_to_center:
            self.flag_center = True
        elif abs(omega) < self.config.center_to_armor:
            self.flag_center = False

        # 获取旋转和射击位姿
        pred_armor_pos = self.track_queue.get_pred_armor_pos(fly_delay + self.rotate_delay)
        pred_robot_pos = None

        if self.flag_antitop:
            if self.flag_center:
                # 中心模式
                pred_armor_pos = self.antitop.get_pred_waiting_mode(fly_delay + self.rotate_delay)
                self.flag_str = "antitop waiting"
                # print("anti-top center")
            else:
                # 装甲板模式
                pred_armor_pos = self.antitop.get_pred_locking_mode(fly_delay + self.rotate_delay)
                self.flag_str = "antitop locking"
                # print("anti-top armor")

            pred_robot_pos = self.antitop.get_pred_robot_pos(fly_delay + self.rotate_delay)
        else:
            self.flag_str = "trackqueue"
            # print("track_queue")
            pass

        pred_pos.append(pred_robot_pos)
        pred_pos.append(pred_armor_pos)

        return pred_pos





