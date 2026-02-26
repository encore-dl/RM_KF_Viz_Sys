import numpy as np
from core.algorithms.filters.extended_kalman import ExtendedKalmanFilter
from models.demo4.utils.my_math import limit_rad

class DemoModel4:
    def __init__(self):
        self.state_dim = 11
        self.ekf = ExtendedKalmanFilter(
            state_dim=self.state_dim,
            x_add_func=lambda x, d: x + d
        )
        self.ekf.P = np.eye(self.state_dim)
        self.ekf.P[6, 6] = 1.0
        self.ekf.P[7, 7] = 200.0
        self.ekf.P[8, 8] = 0.01
        self.ekf.P[9, 9] = 0.01

        q_x, q_y, q_z = 0.01, 0.01, 0.01
        q_vx, q_vy, q_vz = 5.0, 5.0, 5.0
        q_yaw, q_omg = 1e-4, 0.5
        q_ra, q_rb, q_dz = 1e-7, 1e-7, 1e-7
        self.Q = np.diag([q_x, q_y, q_z, q_vx, q_vy, q_vz, q_yaw, q_omg, q_ra, q_rb, q_dz])

        r_x, r_y, r_z, r_yaw = 0.05, 0.05, 0.05, 0.02
        self.R = np.diag([r_x, r_y, r_z, r_yaw])

        self.last_t = 0
        self.is_init = False
        self.match_id = 0
        self.default_r = 0.2
        self.spin_thresh = 2.0

    @staticmethod
    def _meas_sub(z, z_pred):
        diff = z - z_pred
        diff[3] = limit_rad(diff[3])
        return diff

    def init_model(self, obs_pos, obs_yaw, t):
        x0 = np.zeros(self.state_dim)
        x0[0] = obs_pos[0] - self.default_r * np.cos(obs_yaw)
        x0[1] = obs_pos[1] - self.default_r * np.sin(obs_yaw)
        x0[2] = obs_pos[2]
        x0[6] = obs_yaw
        x0[8] = self.default_r
        x0[9] = self.default_r
        self.ekf.x = x0
        self.ekf.P = np.eye(self.state_dim) * 0.1
        self.ekf.P[7, 7] = 50.0
        self.last_t = t
        self.is_init = True

    def predict(self, dt):
        if dt <= 0: return
        def f_func(x):
            nx = x.copy()
            nx[0] += x[3] * dt
            nx[1] += x[4] * dt
            nx[2] += x[5] * dt
            nx[6] += x[7] * dt
            return nx
        def F_jacob(x):
            F = np.eye(self.state_dim)
            F[0, 3] = dt
            F[1, 4] = dt
            F[2, 5] = dt
            F[6, 7] = dt
            return F
        Q_step = self.Q * dt
        self.ekf.predict(F=None, Q=Q_step, f_func=f_func, F_jacobian=F_jacob)

    def update(self, obs_pos, obs_yaw, t):
        if not self.is_init:
            self.init_model(obs_pos, obs_yaw, t)
            return

        dt = t - self.last_t
        self.last_t = t
        # 注意：外部已调用predict，此处不再predict

        pred_psi = self.ekf.x[6]
        self.match_id, aligned_yaw = self._solve_match_id(obs_yaw, pred_psi)
        z_meas = np.array([obs_pos[0], obs_pos[1], obs_pos[2], aligned_yaw])
        is_even = (self.match_id % 2 == 0)

        def h_func(x):
            xc, yc, zc = x[0], x[1], x[2]
            psi = x[6]
            ra, rb, dz = x[8], x[9], x[10]
            r = ra if is_even else rb
            z_offset = 0.0 if is_even else dz
            armor_yaw = psi + self.match_id * (np.pi / 2.0)
            return np.array([
                xc + r * np.cos(armor_yaw),
                yc + r * np.sin(armor_yaw),
                zc + z_offset,
                armor_yaw
            ])

        def H_jacob(x):
            H = np.zeros((4, self.state_dim))
            psi = x[6]
            ra, rb = x[8], x[9]
            r = ra if is_even else rb
            armor_yaw = psi + self.match_id * (np.pi / 2.0)
            c, s = np.cos(armor_yaw), np.sin(armor_yaw)
            H[0, 0] = 1.0
            H[0, 6] = -r * s
            if is_even: H[0, 8] = c
            else: H[0, 9] = c
            H[1, 1] = 1.0
            H[1, 6] = r * c
            if is_even: H[1, 8] = s
            else: H[1, 9] = s
            H[2, 2] = 1.0
            if not is_even: H[2, 10] = 1.0
            H[3, 6] = 1.0
            return H

        self.ekf.update(
            z=z_meas, H=None, R=self.R,
            z_sub_func=self._meas_sub,
            h_func=h_func, H_jacob=H_jacob
        )

    @staticmethod
    def _solve_match_id(obs_yaw, pred_psi):
        best_id = 0
        min_diff = float('inf')
        aligned_yaw = obs_yaw
        for i in range(4):
            raw_diff = (obs_yaw - i * (np.pi / 2.0)) - pred_psi
            diff = limit_rad(raw_diff)
            if abs(diff) < min_diff:
                min_diff = abs(diff)
                best_id = i
                aligned_yaw = pred_psi + diff + i * (np.pi / 2.0)
        return best_id, aligned_yaw

    def get_pred_pos(self, fly_t):
        if not self.is_init:
            return np.zeros(3), np.zeros(3)
        x = self.ekf.x
        dt = fly_t
        pred_cx = x[0] + x[3] * dt
        pred_cy = x[1] + x[4] * dt
        pred_cz = x[2] + x[5] * dt
        psi_pred = x[6] + x[7] * dt
        is_even = (self.match_id % 2 == 0)
        r = x[8] if is_even else x[9]
        dz = 0.0 if is_even else x[10]

        if abs(x[7]) < self.spin_thresh:
            armor_yaw = psi_pred + self.match_id * (np.pi / 2.0)
        else:
            yaw_to_self = np.arctan2(-pred_cy, -pred_cx)
            armor_yaw = yaw_to_self

        pred_armor = [
            pred_cx + r * np.cos(armor_yaw),
            pred_cy + r * np.sin(armor_yaw),
            pred_cz + dz
        ]
        return [pred_cx, pred_cy, pred_cz], pred_armor

    def predict_observation(self, k):
        """返回预测的第k个装甲板的观测向量 [x,y,z,yaw] 及其协方差 S"""
        x = self.ekf.x
        P = self.ekf.P
        is_even = (k % 2 == 0)
        r = x[8] if is_even else x[9]
        dz = x[10] if not is_even else 0.0
        psi = x[6]
        armor_yaw = psi + k * np.pi / 2.0
        z_pred = np.array([
            x[0] + r * np.cos(armor_yaw),
            x[1] + r * np.sin(armor_yaw),
            x[2] + dz,
            armor_yaw
        ])
        H = np.zeros((4, self.state_dim))
        c, s = np.cos(armor_yaw), np.sin(armor_yaw)
        H[0, 0] = 1.0
        H[0, 6] = -r * s
        if is_even:
            H[0, 8] = c
        else:
            H[0, 9] = c
        H[1, 1] = 1.0
        H[1, 6] = r * c
        if is_even:
            H[1, 8] = s
        else:
            H[1, 9] = s
        H[2, 2] = 1.0
        if not is_even: H[2, 10] = 1.0
        H[3, 6] = 1.0
        S = H @ P @ H.T + self.R
        return z_pred, S

    def get_all_armor_positions_at_time(self, dt):
        """预测未来 dt 秒后所有装甲板的世界坐标"""
        if not self.is_init:
            return [np.zeros(3)] * 4
        x = self.ekf.x
        pred_cx = x[0] + x[3] * dt
        pred_cy = x[1] + x[4] * dt
        pred_cz = x[2] + x[5] * dt
        pred_psi = x[6] + x[7] * dt
        ra, rb, dz = x[8], x[9], x[10]

        armors = []
        for k in range(4):
            is_even = (k % 2 == 0)
            r = ra if is_even else rb
            h = 0.0 if is_even else dz
            armor_yaw = pred_psi + k * np.pi / 2.0
            ax = pred_cx + r * np.cos(armor_yaw)
            ay = pred_cy + r * np.sin(armor_yaw)
            az = pred_cz + h
            armors.append(np.array([ax, ay, az]))
        return armors

    def get_geometric_distance(self, obs_pos):
        """计算观测点到四个理论装甲板的最小欧氏距离"""
        x = self.ekf.x
        xc, yc, zc = x[0], x[1], x[2]
        psi = x[6]
        min_dist = float('inf')
        for k in range(4):
            is_even = (k % 2 == 0)
            r = x[8] if is_even else x[9]
            dz = x[10] if not is_even else 0.0
            armor_yaw = psi + k * (np.pi / 2.0)
            pred_ax = xc + r * np.cos(armor_yaw)
            pred_ay = yc + r * np.sin(armor_yaw)
            pred_az = zc + dz
            dist = np.sqrt((obs_pos[0] - pred_ax)**2 +
                           (obs_pos[1] - pred_ay)**2 +
                           (obs_pos[2] - pred_az)**2)
            if dist < min_dist:
                min_dist = dist
        return min_dist



