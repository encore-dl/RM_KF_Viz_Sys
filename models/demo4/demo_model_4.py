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

        # ---------- 过程噪声基值（连续时间密度）----------
        self.q_xy   = 0.01      # 位置噪声密度 (m^2/s)
        self.q_z    = 0.01
        self.q_vxy  = 5.0       # 速度噪声密度 (m^2/s^3)
        self.q_vz   = 5.0
        self.q_yaw  = 1e-4      # 角度噪声密度 (rad^2/s)
        self.q_omg  = 0.5       # 角速度噪声密度 (rad^2/s^3)
        self.q_r    = 1e-7      # 半径变化率密度 (m^2/s)   # 基值，后续动态调整
        self.q_dz   = 1e-7      # 高度差变化率密度 (m^2/s)
        self.Q = np.diag([
            self.q_xy, self.q_xy, self.q_z,
            self.q_vxy, self.q_vxy, self.q_vz,
            self.q_yaw, self.q_omg,
            self.q_r, self.q_r, self.q_dz
        ])

        # ---------- 常值观测噪声 ----------
        self.r_xy  = 0.5
        self.r_z   = 0.5
        self.r_yaw = 0.02
        self.R = np.diag([self.r_xy, self.r_xy, self.r_z, self.r_yaw])

        self.last_t = 0
        self.is_init = False
        self.match_id = 0
        self.default_r = 0.2
        self.spin_thresh = 3.0

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
        if dt <= 0:
            return

        # 离散化：乘以 dt
        Q_step = self.Q * dt

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

        self.ekf.predict(F=None, Q=Q_step, f_func=f_func, F_jacobian=F_jacob)

    def update(self, obs_pos, obs_yaw, t):
        if not self.is_init:
            self.init_model(obs_pos, obs_yaw, t)
            return

        dt = t - self.last_t
        self.last_t = t

        # 保存上一帧匹配的ID用于锁定偏置
        last_id = self.match_id
        pred_psi = self.ekf.x[6]

        # 使用新的匹配方法
        self.match_id, aligned_yaw = self._solve_match_id(obs_pos, obs_yaw, pred_psi, last_id)
        z_meas = np.array([obs_pos[0], obs_pos[1], obs_pos[2], aligned_yaw])
        is_even = (self.match_id % 2 == 0)

        # 直接使用常值观测噪声 R
        R = self.R

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
            z=z_meas, H=None, R=R,
            z_sub_func=self._meas_sub,
            h_func=h_func, H_jacob=H_jacob
        )

    def _solve_match_id(self, obs_pos, obs_yaw, pred_psi, last_id):
        """
        基于马氏距离和概率的装甲板匹配。
        返回 (best_id, aligned_yaw)
        """
        best_id = 0
        max_log_post = -np.inf
        aligned_yaw = obs_yaw

        # 当前角速度（用于动态保持概率）
        omega = abs(self.ekf.x[7])
        # 保持概率：角速度为0时0.95，达自旋阈值时0.6（可调）
        p_keep = max(0.6, min(0.95, 0.95 - 0.1 * omega))

        for i in range(4):
            # 获取预测观测及其协方差
            z_pred, S = self.predict_observation(i)

            # 计算残差，角度分量限幅
            y = np.array([
                obs_pos[0] - z_pred[0],
                obs_pos[1] - z_pred[1],
                obs_pos[2] - z_pred[2],
                limit_rad(obs_yaw - z_pred[3])
            ])

            # 计算马氏距离平方（使用 solve 避免显式求逆，提高数值稳定性）
            try:
                # 求解 S * x = y
                x = np.linalg.solve(S, y)
                d = y @ x  # 等价于 y.T @ inv(S) @ y
            except np.linalg.LinAlgError:
                # 协方差奇异，设为极大值
                d = 1e12

            # 对数似然（省略常数项）
            logL = -0.5 * d

            # 对数先验
            if last_id is None:
                log_prior = np.log(0.25)
            else:
                if i == last_id:
                    log_prior = np.log(p_keep)
                else:
                    log_prior = np.log((1 - p_keep) / 3.0)

            log_post = logL + log_prior

            # 可选的 hysteresis：如果上一帧 ID 的后验不低于最佳后验的某个阈值，可保留上一帧
            # 这里先记录最佳，后面统一处理

            if log_post > max_log_post:
                max_log_post = log_post
                best_id = i
                # 对齐后的观测角度（用于更新）
                aligned_yaw = z_pred[3] + y[3]

        # 可选 hysteresis：如果上一帧 ID 存在，且其后验与最佳后验相差不大，则保持
        if last_id is not None:
            # 重新计算上一帧的后验（为效率可缓存，但简单起见重新计算一次）
            z_pred_last, S_last = self.predict_observation(last_id)
            y_last = np.array([
                obs_pos[0] - z_pred_last[0],
                obs_pos[1] - z_pred_last[1],
                obs_pos[2] - z_pred_last[2],
                limit_rad(obs_yaw - z_pred_last[3])
            ])
            try:
                x_last = np.linalg.solve(S_last, y_last)
                d_last = y_last @ x_last
                logL_last = -0.5 * d_last
                # 先验中保持概率已包含，这里不再重复，直接比较后验
                # 实际上上一帧后验已在循环中计算过，但为了简化，我们比较对数似然差
                # 如果上一帧的对数似然不低于最佳对数似然减去阈值，则保持
                if logL_last >= max_log_post - 1.0:  # 阈值可调
                    best_id = last_id
                    aligned_yaw = z_pred_last[3] + y_last[3]
            except:
                pass

        return best_id, aligned_yaw

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