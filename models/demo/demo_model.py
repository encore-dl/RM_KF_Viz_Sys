import time

import numpy as np

from core.algorithms.filters.kalman import KalmanFilter
from core.algorithms.filters.unscented_kalman import UnscentedKalmanFilter
from models.demo.utils.my_math import (
    limit_rad
)


class DemoModel:
    def __init__(self):
        # motion
        self.motion_state_dim = 8

        self.kf_motion = KalmanFilter(state_dim=self.motion_state_dim)
        self.kf_motion.P = np.eye(self.motion_state_dim) * 1.
        self.Q_motion = np.diag([
            0.05, 0.05, 0.001,  # pos
            0.20, 0.20, 0.20,  # vel
            50., 50.  # acc (加速度突变大)
        ])
        self.R_motion = np.diag([
            0.01, 0.01, 0.01
        ])

        # rotation
        self.rot_state_dim = 6

        self.ukf_rot = UnscentedKalmanFilter(
            state_dim=self.rot_state_dim,
            alpha=0.1, beta=2., kappa=0.,
            x_add_func=self._rot_x_add,
            z_sub_func=self._rot_z_sub
        )
        self.ukf_rot.P = np.diag([
            1.0, 1.0, 1.0, 100.0, 100.0, 1.0
        ])
        self.Q_rot_trans = np.diag([
            1e-5, 1e-4, 0.01,    # psi, omega, beta (允许角度微调)
            1e-8, 1e-8, 1e-8     # r_a, r_b, dz (死锁结构参数！)
        ])
        self.Q_rot_spining = np.diag([
            1e-4, 0.05, 1.0,  # psi, omega, beta
            0.01, 0.01, 1e-5  # r_a, r_b, dz (结构参数相对稳定)
        ])
        self.R_rot = np.diag([
            0.02, 0.05, 0.05
        ])

        self.last_t = time.time()
        self.is_init = False
        self.is_spining = False
        self.match_id = 0

        self.default_r = 0.2
        self.F_a_dec = 0.6
        self.spin_thresh = 2.
        self.spin_obsrv_thresh = 5.

    @staticmethod
    def _rot_x_add(a, b):
        return a + b

    @staticmethod
    def _rot_z_sub(z, z_pred):
        diff = z - z_pred
        diff[0] = limit_rad(diff[0])
        return diff

    def init_model(self, obs_pos, obs_yaw, t):
        self.ukf_rot.x = np.array([
            obs_yaw, 0, 0,  # psi, w, b
            self.default_r, self.default_r, 0.0  # ra, rb, dz
        ])

        self.kf_motion.x = np.zeros(self.motion_state_dim)
        self.kf_motion.x[:3] = np.array([
            obs_pos[0] - self.default_r * np.cos(obs_yaw),
            obs_pos[1] - self.default_r * np.sin(obs_yaw),
            obs_pos[2]  # z轴暂时不管
        ])
        # --- 关键修改结束 ---

        # 配合调参：初始化时给予极大的不确定度 P
        self.ukf_rot.P = np.diag([
            1, 1, 1, 100, 100, 1
        ])
        self.kf_motion.P = np.eye(8) * 10.0  # 运动 P 也给大点

        self.last_t = t
        self.is_init = True

    def predict(self, dt):
        if dt <= 0:
            return

        Q_rot_raw = self.Q_rot_spining

        # motion
        # x: [x, y, z, vx, vy, vz, ax, ay]
        F_motion = np.array([
            [1., 0., 0., dt, 0., 0., 0.5*dt**2, 0.],
            [0., 1., 0., 0., dt, 0., 0., 0.5*dt**2],
            [0., 0., 1., 0., 0., dt, 0., 0.],
            [0., 0., 0., 1., 0., 0., dt, 0.],
            [0., 0., 0., 0., 1., 0., 0., dt],
            [0., 0., 0., 0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 0., 0., self.F_a_dec, 0.],
            [0., 0., 0., 0., 0., 0., 0., self.F_a_dec]
        ])
        Q_motion = self.Q_motion * dt
        self.kf_motion.predict(F=F_motion, Q=Q_motion)

        # rot
        def f_rot(x):
            # x: [psi(yaw), w, alpha, ra, rb, dz]
            return np.array([
                x[0] + x[1]*dt + 0.5*x[2]*dt**2,
                x[1] + x[2]*dt,
                x[2] * 0.95,  # 角加速度衰减
                x[3],
                x[4],
                x[5]
            ])
        Q_rot = Q_rot_raw * dt
        self.ukf_rot.predict(Q=Q_rot, f_func=f_rot)

    def update(self, obs_pos, obs_yaw, t):
        if not self.is_init:
            self.init_model(obs_pos, obs_yaw, t)
            return

        dt = t - self.last_t
        self.last_t = t

        self.predict(dt)
        # 上面搬地方

        pred_center = self.kf_motion.x[:3]
        pred_yaw = self.ukf_rot.x[0]

        dist_xy = np.linalg.norm(obs_pos[:2] - pred_center[:2])
        diff_z = obs_pos[2] - pred_center[2]

        self.match_id, aligned_yaw = self._solve_match_id(obs_yaw, pred_yaw)

        z_rot = np.array([
            aligned_yaw,
            dist_xy,
            diff_z
        ])

        def h_rot(x):
            is_even = (self.match_id % 2 == 0)
            r_pred = x[3] if is_even else x[4]
            z_pred = 0. if is_even else x[5]

            return np.array([x[0], r_pred, z_pred])

        self.ukf_rot.update(
            z=z_rot,
            R=self.R_rot,
            h_func=h_rot
        )

        state_rot = self.ukf_rot.x
        r_opt = state_rot[3] if (self.match_id % 2 == 0) else state_rot[4]
        dz_opt = 0.0 if (self.match_id % 2 == 0) else state_rot[5]

        smooth_armor_yaw = state_rot[0] + self.match_id * (np.pi / 2.0)

        v_center_x = obs_pos[0] - r_opt * np.cos(smooth_armor_yaw)
        v_center_y = obs_pos[1] - r_opt * np.sin(smooth_armor_yaw)
        v_center_z = obs_pos[2] - dz_opt

        z_motion = np.array([v_center_x, v_center_y, v_center_z])

        H_motion = np.eye(3, 8)  # 只观测 x, y, z
        self.kf_motion.update(z_motion, H_motion, self.R_motion)

    @staticmethod
    def _solve_match_id(obs_yaw, pred_yaw):
        best_id = 0
        min_diff = float('inf')
        aligned_yaw = 0.

        for i in range(4):
            raw = obs_yaw - i * (np.pi / 2.)
            diff = raw - pred_yaw
            diff = limit_rad(diff)

            dist = abs(diff)
            if dist < min_diff:
                min_diff = dist
                best_id = i
                aligned_yaw = pred_yaw + diff

        return best_id, aligned_yaw

    def get_pred_pos(self, fly_t):
        x_motion = self.kf_motion.x
        x_rot = self.ukf_rot.x
        dt = fly_t

        pred_cx = x_motion[0] + x_motion[3]*dt + 0.5*x_motion[6]*(dt**2)
        pred_cy = x_motion[1] + x_motion[4]*dt + 0.5*x_motion[7]*(dt**2)
        pred_cz = x_motion[2] + x_motion[5]*dt + 0.5*x_rot[5]
        pred_center = [
            pred_cx,
            pred_cy,
            pred_cz
        ]

        omg = self.ukf_rot.x[1]

        if abs(omg) < self.spin_thresh:
            pred_yaw = x_rot[0] + x_rot[1]*dt + 0.5*x_rot[2]*(dt**2)

            tar_id = self.match_id
            is_even = (tar_id % 2 == 0)

            r_pred = x_rot[3] if is_even else x_rot[4]
            dz_pred = 0. if is_even else x_rot[5]

            tar_yaw = pred_yaw + tar_id * (np.pi / 2.)

            pred_armor = [
                pred_center[0] + r_pred * np.cos(tar_yaw),
                pred_center[1] + r_pred * np.sin(tar_yaw),
                pred_center[2] + dz_pred - 0.5*x_rot[5]
            ]

            return [pred_center, pred_armor]
        else:
            yaw_to_self = np.atan2(-pred_cy, -pred_cx)

            tar_id = self.match_id
            is_even = (tar_id % 2 == 0)

            r_pred = x_rot[3] if is_even else x_rot[4]
            dz_pred = 0. if is_even else x_rot[5]

            pred_armor = [
                pred_center[0] + r_pred * np.cos(yaw_to_self),
                pred_center[1] + r_pred * np.sin(yaw_to_self),
                pred_center[2] + dz_pred - 0.5*x_rot[5]
            ]

            return [pred_center, pred_armor]




