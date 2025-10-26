import numpy as np
import math

from object.entity.robot import RobotType
from algorithms.filter.extended_kalman_filter import ExtendedKalmanFilter
from utils.math_tool import limit_rad, pos_to_tpd, pos_to_tpd_jacob


# 状态向量元素: [x, vx, y, vy, z, vz, agl, omg, r, dr, dz] 11维向量
# 观测向量元素: [roll, pitch, yaw, agl]
class TongJiModel:
    def __init__(self):
        self.armor = None
        self._ekf = None

        self.last_id = None
        self.armor_count = 0

        self.update_count = 0

        self.is_converged = False

    def init_model(self, armor, armor_count):
        self.armor = armor
        self.armor_count = armor_count

        x0 = np.array([
            self.armor.world_pos[0], 0,
            self.armor.world_pos[1], 0,
            self.armor.world_pos[2], 0,
            self.armor.world_rpy[0], 0,
            self.armor.radius, 0, 0,
        ])
        P0_diag = np.array([1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1])
        P0 = np.diag(P0_diag)

        def x_add(a, b):
            c = a + b
            c[6] = limit_rad(c[6])
            return c

        self._ekf = ExtendedKalmanFilter(x0, P0, x_add)

    def predict(self, dt):
        F = np.eye(11)
        F[0, 1] = dt  # x = x + vx*dt
        F[2, 3] = dt  # y = y + vy*dt
        F[4, 5] = dt  # z = z + vz*dt
        F[6, 7] = dt  # a = a + w*dt

        v1, v2 = (10, 0.1) if self.armor.robot_type == RobotType.Outpost else (100, 400)
        p_noise = dt ** 4 / 4
        pv_noise = dt ** 3 / 2
        v_noise = dt ** 2
        Q = np.zeros((11, 11))
        Q[0:2, 0:2] = [  # x, vx
            [p_noise * v1, pv_noise * v1],
            [pv_noise * v1, v_noise * v1]
        ]
        Q[2:4, 2:4] = [  # y, vy
            [p_noise * v1, pv_noise * v1],
            [pv_noise * v1, v_noise * v1]
        ]
        Q[4:6, 4:6] = [  # z, vz
            [p_noise * v1, pv_noise * v1],
            [pv_noise * v1, v_noise * v1]
        ]
        Q[6:8, 6:8] = [  # agl, omg
            [p_noise * v2, pv_noise * v2],
            [pv_noise * v2, v_noise * v2]
        ]

        # 人工设置前哨战的最高转速
        if self.is_converged and self.armor.robot_type == RobotType.Outpost and abs(self._ekf.x[7]) > 2:
            self._ekf.x[7] = 2.51 if self._ekf.x[7] > 0 else -2.51

        self._ekf.predict(F, Q)

        self._ekf.x[6] = limit_rad(self._ekf.x[6])

    def get_est_armor_pos(self, x_, armor_id):
        est_armor_agl = limit_rad(x_[6] + armor_id * 2 * math.pi / self.armor_count)
        is_change_l_h = (self.armor_count == 4) and (armor_id % 2 == 1)

        r = x_[8] + x_[9] if is_change_l_h else x_[8]
        z_offs = x_[10] if is_change_l_h else 0.

        armor_x = x_[0] + r * math.cos(est_armor_agl)
        armor_y = x_[2] + r * math.sin(est_armor_agl)
        armor_z = x_[4] + z_offs

        return np.array([armor_x, armor_y, armor_z])

    def update(self, armor):
        x = self._ekf.x
        est_armor_agl = limit_rad(x[6] + armor.armor_id * 2 * math.pi / self.armor_count)
        is_change_l_h = (self.armor_count == 4) and (armor.armor_id % 2 == 1)

        r = x[8] + x[9] if is_change_l_h else x[8]
        z_offs = x[10] if is_change_l_h else 0.

        dx_da = -r * math.sin(est_armor_agl)  # x 对 agl 求偏导
        dy_da = r * math.cos(est_armor_agl)

        dx_dr = math.cos(est_armor_agl)
        dy_dr = math.sin(est_armor_agl)

        dx_dl = math.cos(est_armor_agl) if is_change_l_h else 0.
        dy_dl = math.sin(est_armor_agl) if is_change_l_h else 0.

        dz_dh = 1. if is_change_l_h else 0.

        def get_est_armor_pos(x_):
            return self.get_est_armor_pos(x_, armor.armor_id)

        est_armor_pos = get_est_armor_pos(x)
        est_armor_tpd = pos_to_tpd_jacob(est_armor_pos)

        H_armor_xyza = np.array([
            [1, 0, 0, 0, 0, 0, dx_da, 0, dx_dr, dx_dl, 0],
            [0, 0, 1, 0, 0, 0, dy_da, 0, dy_dr, dy_dl, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, dz_dh],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        ])

        H_armor_tpda = np.array([
            [est_armor_tpd[0, 0], est_armor_tpd[0, 1], est_armor_tpd[0, 2], 0],
            [est_armor_tpd[1, 0], est_armor_tpd[1, 1], est_armor_tpd[1, 2], 0],
            [est_armor_tpd[2, 0], est_armor_tpd[2, 1], est_armor_tpd[2, 2], 0],
            [0, 0, 0, 1]
        ])

        # 观测雅可比矩阵
        H = H_armor_tpda @ H_armor_xyza

        # 自适应观测噪声
        center_yaw = math.atan2(armor.world_pos[1], armor.world_pos[0])
        delta_agl = limit_rad(armor.world_rpy[2] - center_yaw)

        R_diag = np.array([
            4e-3,  # yaw noise
            4e-3,  # pitch noise
            math.log(abs(delta_agl) + 1) + 1,  # distance noise
            math.log(abs(armor.world_rpy[0]) + 1) / 200 + 9e-2  # agl noise
        ])
        R = np.diag(R_diag)

        def h_func(x_):
            pos = get_est_armor_pos(x_)
            tpd = pos_to_tpd(pos)
            agl = est_armor_agl

            return np.array([tpd[0], tpd[1], tpd[2], agl])

        z = np.array([
            armor.world_tpd[0],
            armor.world_tpd[1],
            armor.world_tpd[2],
            armor.world_rpy[2]
        ])

        def z_subtract(a, b):
            c = a - b
            c[0] = limit_rad(c[0])  # yaw
            c[1] = limit_rad(c[1])  # pitch
            c[3] = limit_rad(c[3])  # agl
            return c

        self._ekf.update(z=z, H=H, R=R, h_func=h_func, z_subtract_func=z_subtract)

        # 同济这有一个 match armor 的操作 以及 记录装甲板旋转数

        self.last_id = armor.armor_id
        self.update_count += 1

    def diverged(self):
        r_ok = 0.05 < self._ekf.x[8] < 0.5
        l_ok = 0.05 < (self._ekf.x[8] + self._ekf.x[9]) < 0.5

        return not (r_ok and l_ok)

    def converged(self):
        if self.armor.robot_type != RobotType.Outpost and self.update_count > 3 and not self.diverged():
            self.is_converged = True
        if self.armor.robot_type == RobotType.Outpost and self.update_count > 10 and not self.diverged():
            self.is_converged = True

        return self.is_converged

    def get_ekf(self):
        return self._ekf









