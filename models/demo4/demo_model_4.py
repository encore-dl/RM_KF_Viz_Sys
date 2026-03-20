import numpy as np

from core.entities.property.robot_type import RobotType
from core.algorithms.filters.extended_kalman import ExtendedKalmanFilter
from core.algorithms.filters.kalman import KalmanFilter
from models.demo4.utils.my_math import limit_rad


class DemoModel4:
    def __init__(self):
        self.robot_type = None
        self.state_dim = 12
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
        self.q_yaw  = 0.1      # 角度噪声密度 (rad^2/s)
        self.q_omg  = 0.5       # 角速度噪声密度 (rad^2/s^3)
        self.q_r    = 1e-7      # 半径变化率密度 (m^2/s)   # 基值，后续动态调整
        self.q_dz   = 1e-7      # 高度差变化率密度 (m^2/s)
        self.Q = np.diag([
            self.q_xy, self.q_xy, self.q_z,
            self.q_vxy, self.q_vxy, self.q_vz,
            self.q_yaw, self.q_omg,
            self.q_r, self.q_r,
            self.q_dz, self.q_dz
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

        self.mahal_thresh_ = 9.49  # 马氏距离平方阈值（4维卡方95%分位）
        self.confirm_thresh_ = 3  # 新ID需要连续匹配成功多少帧才允许切换
        self.max_lost_ = 2  # 连续丢失多少帧后重置计数
        # 为每个装甲板ID维护状态
        armor_count = 4  # 默认4，实际会在init_model中根据车型更新
        self.id_counts_ = [0] * armor_count  # 连续匹配成功次数
        self.id_lost_ = [0] * armor_count  # 连续丢失次数
        self.stable_id_ = 0  # 当前稳定输出的ID

        #
        self.cont_yaw = 0.
        self.last_obs_yaw = 0.
        self.angle_kf = KalmanFilter(state_dim=2, x_add_func=lambda a,b:a+b)

    @staticmethod
    def _meas_sub(z, z_pred):
        diff = z - z_pred
        diff[3] = limit_rad(diff[3])
        return diff

    def init_model(self, obs):
        self.robot_type = obs.robot_type
        self.is_init = True
        obs_pos = obs.rel_pos
        obs_yaw = obs.rel_rpy[2]

        #
        self.cont_yaw = obs_yaw
        self.last_obs_yaw = obs_yaw
        self.angle_kf.x = np.array([obs_yaw, 0.])
        self.angle_kf.P = np.eye(2) * 0.1

        x0 = np.zeros(self.state_dim)
        x0[0] = obs_pos[0] - self.default_r * np.cos(obs_yaw)
        x0[1] = obs_pos[1] - self.default_r * np.sin(obs_yaw)
        x0[2] = obs_pos[2]
        x0[6] = obs_yaw
        x0[8] = self.default_r
        x0[9] = self.default_r
        x0[10] = 0.
        x0[11] = 0.
        self.ekf.x = x0
        self.ekf.P = np.eye(self.state_dim) * 0.1
        self.ekf.P[7, 7] = 50.0

        armor_count = self._get_armor_count()
        self.id_counts_ = [0] * armor_count
        self.id_lost_ = [0] * armor_count
        self.stable_id_ = 0
        self.match_id = 0

    def predict(self, dt):
        if dt <= 0:
            return

        #
        F_angle = np.array([
            [1, dt],
            [0, 1]
        ])
        Q_angle = np.diag([0.01*dt, 0.1*dt])
        self.angle_kf.predict(F=F_angle, Q=Q_angle)

        # 主模型进行预测
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

    def update(self, obs):
        if not self.is_init or obs.robot_type != self.robot_type:
            self.init_model(obs)
            return

        obs_pos = obs.rel_pos
        obs_yaw = obs.rel_rpy[2]

        #
        angle_diff = limit_rad(obs_yaw - self.last_obs_yaw)
        if abs(angle_diff) > np.pi/4:
            corr = np.pi/2 if angle_diff > 0 else -np.pi/2
            angle_diff -= corr
        self.cont_yaw += angle_diff
        self.last_obs_yaw = obs_yaw

        H_angle = np.array([[1, 0]])
        R_angle = np.eye(1) * 0.1
        z_angle = np.array([self.cont_yaw])
        def z_sub_angle(z, z_pred):
            diff = z - z_pred
            diff[0] = limit_rad(diff[0])
            return diff
        self.angle_kf.update(z_angle, H_angle, R_angle, z_sub_func=z_sub_angle)

        print(f"AngleKF omega: {self.angle_kf.x[1]:.6f}")
        print(f"AngleKF theta: {self.angle_kf.x[0]:.6f}")
        print()

        # 主模型更新
        self.match_id, aligned_yaw = self._solve_match_id(obs_pos, obs_yaw)
        z_meas = np.array([obs_pos[0], obs_pos[1], obs_pos[2], aligned_yaw])

        R = self.R

        def h_func(x):
            xc, yc, zc = x[0], x[1], x[2]
            psi = x[6]
            r, dz, _, _ = self._get_armor_params(self.match_id)
            armor_yaw = psi + self.match_id * (np.pi / 2.0)
            return np.array([
                xc + r * np.cos(armor_yaw),
                yc + r * np.sin(armor_yaw),
                zc + dz,
                armor_yaw
            ])

        def H_jacob(x):
            H = np.zeros((4, self.state_dim))
            psi = x[6]
            r, dz, r_idx, dz_idx = self._get_armor_params(self.match_id)
            armor_yaw = psi + self.match_id * (np.pi / 2.0)
            c, s = np.cos(armor_yaw), np.sin(armor_yaw)

            H[0, 0] = 1.0
            H[0, 6] = -r * s
            if r_idx is not None:
                H[0, r_idx] = c
            H[1, 1] = 1.0
            H[1, 6] = r * c
            if r_idx is not None:
                H[1, r_idx] = s
            H[2, 2] = 1.0
            if dz_idx is not None:
                H[2, dz_idx] = 1.0
            H[3, 6] = 1.0
            return H

        self.ekf.update(
            z=z_meas, H=None, R=R,
            z_sub_func=self._meas_sub,
            h_func=h_func, H_jacob=H_jacob
        )

    def _get_armor_params(self, k):
        x = self.ekf.x
        robot_type = self.robot_type

        if robot_type == RobotType.Outpost:
            if k == 0:
                r = x[8]
                dz = 0.
                r_idx = 8
                dz_idx = None
            elif k == 1:
                r = x[8]
                dz = x[10]
                r_idx = 8
                dz_idx = 10
            else:
                r = x[8]
                dz = x[11]
                r_idx = 8
                dz_idx = 11
        # elif robot_type == RobotType.Sentry:
        #     r = x[8]
        #     dz = 0.
        #     r_idx = 8
        #     dz_idx = None
        else:
            if k % 2 == 0:
                r = x[8]
                dz = 0.
                r_idx = 8
                dz_idx = None
            else:
                r = x[9]
                dz = x[10]
                r_idx = 9
                dz_idx = 10

        return r, dz, r_idx, dz_idx

    def _solve_match_id(self, obs_pos, obs_yaw):
        armor_count = self._get_armor_count()
        # 确保状态列表长度正确
        if len(self.id_counts_) != armor_count:
            self.id_counts_ = [0] * armor_count
            self.id_lost_ = [0] * armor_count

        # 构建完整观测向量
        obs = np.array([obs_pos[0], obs_pos[1], obs_pos[2], obs_yaw])

        match_success = [False] * armor_count
        match_dist = [float('inf')] * armor_count  # 存储马氏距离平方

        # 1. 计算每个ID的马氏距离，判断是否匹配成功
        for i in range(armor_count):
            z_pred, S = self.predict_observation(i)
            y = obs - z_pred
            y[3] = limit_rad(y[3])  # 角度差归一化

            try:
                # 求解马氏距离平方 d = y^T * S^{-1} * y
                # 使用 np.linalg.solve 更稳定
                x = np.linalg.solve(S, y)
                d = y @ x
            except np.linalg.LinAlgError:
                d = float('inf')

            match_dist[i] = d
            if d < self.mahal_thresh_:
                match_success[i] = True

        # 2. 更新各ID的计数和丢失状态
        for i in range(armor_count):
            if match_success[i]:
                self.id_counts_[i] += 1
                self.id_lost_[i] = 0
            else:
                self.id_lost_[i] += 1
                if self.id_lost_[i] > self.max_lost_:
                    self.id_counts_[i] = 0  # 重置计数

        # 3. 确定最终ID
        best_id = self.stable_id_  # 默认用上一个稳定ID
        # 检查上一稳定ID是否仍然有效（匹配成功且计数≥1）
        if match_success[self.stable_id_] and self.id_counts_[self.stable_id_] > 0:
            # 上一ID仍有效，保持不变
            pass
        else:
            # 从所有ID中选择计数最高的作为新候选
            max_count = -1
            candidate_id = -1
            for i in range(armor_count):
                if match_success[i] and self.id_counts_[i] > max_count:
                    max_count = self.id_counts_[i]
                    candidate_id = i
            if candidate_id != -1 and max_count >= self.confirm_thresh_:
                # 候选ID连续计数达到确认阈值，允许切换
                best_id = candidate_id
            elif candidate_id != -1:
                # 候选ID未达到确认阈值，但上一ID已失效，暂用候选（后续会因计数不足被过滤）
                best_id = candidate_id
            else:
                # 无任何有效匹配，降级：选择马氏距离最小的ID
                best_id = int(np.argmin(match_dist))

        # 更新稳定ID
        self.stable_id_ = best_id

        # 4. 计算对齐后的yaw
        z_pred_final, _ = self.predict_observation(best_id)
        final_pred_yaw = z_pred_final[3]
        yaw_residual = limit_rad(obs_yaw - final_pred_yaw)
        aligned_yaw = final_pred_yaw + yaw_residual

        return best_id, aligned_yaw

    def _get_armor_count(self):
        if self.robot_type == RobotType.Outpost:
            return 3
        # elif self.robot_type == RobotType.Sentry:
        #     return 2
        else:
            return 4

    def predict_observation(self, k):
        """返回预测的第k个装甲板的观测向量 [x,y,z,yaw] 及其协方差 S"""
        x = self.ekf.x
        P = self.ekf.P
        r, dz, r_idx, dz_idx = self._get_armor_params(k)
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
        if r_idx is not None:
            H[0, r_idx] = c
        H[1, 1] = 1.0
        H[1, 6] = r * c
        if r_idx is not None:
            H[1, r_idx] = s
        H[2, 2] = 1.0
        if dz_idx is not None:
            H[2, dz_idx] = 1.0
        H[3, 6] = 1.0
        S = H @ P @ H.T + self.R
        return z_pred, S

    def get_all_armor_positions_at_time(self, dt):
        if not self.is_init:
            return [np.zeros(3)] * (4 if self.robot_type is None else self._get_armor_count())

        x = self.ekf.x
        pred_cx = x[0] + x[3] * dt
        pred_cy = x[1] + x[4] * dt
        pred_cz = x[2] + x[5] * dt
        pred_psi = x[6] + x[7] * dt

        armor_count = self._get_armor_count()
        armors = []
        for k in range(armor_count):
            r, dz, _, _ = self._get_armor_params(k)
            armor_yaw = pred_psi + k * np.pi / 2.0
            ax = pred_cx + r * np.cos(armor_yaw)
            ay = pred_cy + r * np.sin(armor_yaw)
            az = pred_cz + dz
            armors.append(np.array([ax, ay, az]))
        return armors

    def get_geometric_distance(self, obs_pos):
        x = self.ekf.x
        xc, yc, zc = x[0], x[1], x[2]
        psi = x[6]

        armor_count = self._get_armor_count()
        min_dist = float('inf')
        for k in range(armor_count):
            r, dz, _, _ = self._get_armor_params(k)
            armor_yaw = psi + k * np.pi / 2.0
            pred_ax = xc + r * np.cos(armor_yaw)
            pred_ay = yc + r * np.sin(armor_yaw)
            pred_az = zc + dz
            dist = np.sqrt((obs_pos[0] - pred_ax) ** 2 +
                           (obs_pos[1] - pred_ay) ** 2 +
                           (obs_pos[2] - pred_az) ** 2)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def diverged(self):
        if not self.is_init:
            return True

        x = self.ekf.x
        ra = x[8]
        if ra < 0.1 or ra > 0.5:
            return True
        rb = x[9]
        if rb < 0.1 or rb > 0.5:
            return True

        if abs(x[3]) > 10 or abs(x[4]) > 10:
            return True

        if self.ekf.get_nis_fail_rate() > self.ekf.nis_threshold:
            return True

        return False







