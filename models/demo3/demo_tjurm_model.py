import time

import numpy as np

from models.demo3.utils.config import TJURMConfig
from models.demo3.submodel.track_queue import TrackQueue
from models.demo3.submodel.demo_model_3 import DemoModel3


class DemoTJURMModel:
    """TJURM主模型，整合TrackQueue和Antitop"""

    def __init__(self):
        self.config = TJURMConfig()

        self.track_queue = TrackQueue(
            self.config.track_count,
            self.config.track_distance,
            self.config.track_delay,
            self.config.track_fire_interval,
            self.config.track_fire_high_delay,
        )
        self.antitop = DemoModel3()

        self.flag_antitop = False
        self.last_armor_id = None
        self.rotate_delay = self.config.rotate_delay
        self.antitop_timeout = 0.5

    def push(self, obs_armor, t):
        pose = np.array([
            obs_armor.world_pos[0],
            obs_armor.world_pos[1],
            obs_armor.world_pos[2],
            obs_armor.world_rpy[2]
        ])
        t = t

        self.track_queue.push(pose, t)

    def update(self):
        self.track_queue.update()

        success, pose, t = self.track_queue.get_antitop_input()

        if not success:
            curr_t = time.time()
            if self.antitop.is_init and curr_t - self.antitop.last_t > self.antitop_timeout:
                self.antitop.is_init = False

            return

        pos = pose[:3]
        yaw = pose[3]

        if not self.antitop.is_init:
            self.antitop.init_model(pos, yaw, t)
        else:
            dt = t - self.antitop.last_t
            if 0 < dt < 1.0:
                self.antitop.update(pos, yaw, t)
            else:
                self.antitop.is_init = False
                self.antitop.init_model(pos, yaw, t)

    def get_pred_pos(self):
        fly_delay = 0.1
        pred_pos = []

        # 角速度判断
        omg = self.antitop.get_omg()
        if abs(omg) > self.config.track_to_antitop:
            self.flag_antitop = True
        elif abs(omg) < self.config.antitop_to_track:
            self.flag_antitop = False

        # 获取旋转和射击位姿
        pred_robot_pos = None
        pred_armor_pos = self.track_queue.get_pred_armor_pos(fly_delay + self.rotate_delay)

        if self.flag_antitop:
            pred_robot_pos, pred_armor_pos = self.antitop.get_pred_pos(fly_delay + self.rotate_delay)
        else:
            # print("track_queue")
            pass

        pred_pos.append(pred_robot_pos)
        pred_pos.append(pred_armor_pos)

        return pred_pos

    # def get_pred_trajectory(self, horizon_steps=50, dt=0.01):
    #     trajectory = []
    #
    #     # 1. 获取当前连续状态 (来自 EKF)
    #     current_psi = self.antitop.ekf.x[6]  # 连续的机器人角度
    #     current_omega = self.antitop.ekf.x[7]  # 角速度
    #     current_id = self.match_id
    #
    #     # 2. 循环生成未来 N 步
    #     for i in range(horizon_steps):
    #         t_future = i * dt
    #
    #         # A. 外推连续角度 (这是平滑的)
    #         pred_psi = current_psi + current_omega * t_future
    #
    #         # B. 几何逻辑判断 (这是跳变来源！)
    #         # 假设边界是 45度 (pi/4)
    #         # 计算当前 ID 下，装甲板的观测角
    #         obs_yaw = pred_psi + current_id * (np.pi / 2) - camera_yaw
    #
    #         # 判断是否越界 (这就是 Psi-d 理论的应用)
    #         if abs(obs_yaw) > VISUAL_LIMIT_ANGLE:
    #             # 触发跳变逻辑！切换到下一块板
    #             current_id = (current_id + 1) % 4
    #
    #         # C. 生成这一步的参考目标
    #         target_yaw = pred_psi + current_id * (np.pi / 2)
    #
    #         trajectory.append(target_yaw)
    #
    #     return trajectory





