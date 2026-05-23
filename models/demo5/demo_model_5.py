import numpy as np
from core.entities.property.robot_type import RobotType
from core.algorithms.filters.extended_kalman import ExtendedKalmanFilter
from core.algorithms.filters.kalman import KalmanFilter
from core.algorithms.math.angle import limit_rad, safe_angle_sub


class DemoModel5:
    """
    完全对齐 antitopV5.cpp 的实现，适配 Python EKF/KF 接口。
    状态向量 (12维): [x, y, z, vx, vy, vz, psi, omega, ra, rb, dz1, dz2]
    - ra: 偶数装甲板半径 (ID 0,2)
    - rb: 奇数装甲板半径 (ID 1,3)
    - dz1: 奇数装甲板高度差 (ID 1)
    - dz2: 备用高度差 (ID 3)，普通车一般不用
    """
    def __init__(self):
        self.robot_type = None
        self.state_dim = 12
        self.ekf = ExtendedKalmanFilter(
            state_dim=self.state_dim,
            x_add_func=lambda x, d: x + d
        )
        # 初始化协方差矩阵 (对应 C++ 构造函数中的 P 初始化)
        self.ekf.P = np.eye(self.state_dim)
        self.ekf.P[6, 6] = 1.0
        self.ekf.P[7, 7] = 200.0
        self.ekf.P[8, 8] = 0.01
        self.ekf.P[9, 9] = 0.01

        # 连续时间过程噪声密度 (对应 setQ 的参数)
        self.q_xy   = 0.01
        self.q_z    = 0.1
        self.q_vxy  = 0.5
        self.q_vz   = 0.5
        self.q_yaw  = 1e-5
        self.q_omg  = 0.2
        self.q_r    = 1e-7
        self.q_dz   = 1e-7
        self.Q_cont = np.diag([
            self.q_xy, self.q_xy, self.q_z,
            self.q_vxy, self.q_vxy, self.q_vz,
            self.q_yaw, self.q_omg,
            self.q_r, self.q_r,
            self.q_dz, self.q_dz
        ])

        # 观测噪声矩阵 (对应 setR)
        self.r_xy  = 0.1
        self.r_z   = 0.1
        self.r_yaw = 0.1
        self.R = np.diag([self.r_xy, self.r_xy, self.r_z, self.r_yaw])

        # 角度辅助卡尔曼滤波器 (对应 angle_kf_)
        self.angle_kf = KalmanFilter(state_dim=2, x_add_func=lambda a, b: a + b)
        self.angle_kf.P = np.eye(2) * 0.1
        # 角度 KF 的过程噪声和观测噪声 (在 predict/update 时传入)
        self.angle_Q_cont = np.diag([0.01, 0.1])
        self.angle_R = np.eye(1) * 0.1

        # 连续角度维护
        self.cont_yaw = 0.0
        self.last_obs_yaw = 0.0

        self.default_radius = 0.2
        self.is_init = False

        # ID 管理
        self.armor_count = 4
        self.id_counts = []
        self.id_lost = []
        self.stable_id = 0
        self.match_id = 0

        # 阈值
        self.dist_thresh = 9.49       # 马氏距离平方阈值 (保留但 solveMatchID 未用)
        self.confirm_thresh = 3
        self.max_lost = 10

        # 滞环参数 (solveMatchID 中用)
        self.keep_threshold = 0.6
        self.hysteresis_bias = 0.45

    @staticmethod
    def _meas_sub(z, z_pred):
        diff = z - z_pred
        diff[3] = limit_rad(diff[3])
        return diff

    def init_model(self, obs):
        """对应 AntitopV5::init"""
        self.robot_type = obs.robot_type
        self.is_init = True
        obs_pos = obs.rel_pos
        obs_yaw = obs.rel_rpy[2]

        # 连续角度初始化
        self.cont_yaw = obs_yaw
        self.last_obs_yaw = obs_yaw
        self.angle_kf.x = np.array([obs_yaw, 0.0])
        self.angle_kf.P = np.eye(2) * 0.1

        # 状态向量初始化
        x0 = np.zeros(self.state_dim)
        x0[0] = obs_pos[0] - self.default_radius * np.cos(obs_yaw)
        x0[1] = obs_pos[1] - self.default_radius * np.sin(obs_yaw)
        x0[2] = obs_pos[2]
        x0[6] = obs_yaw
        if self.robot_type == RobotType.Outpost:
            x0[8] = 0.55   # 前哨站半径固定
        else:
            x0[8] = self.default_radius
            x0[9] = self.default_radius
        x0[10] = 0.0
        x0[11] = 0.0
        self.ekf.x = x0

        self.ekf.P = np.eye(self.state_dim) * 0.1
        self.ekf.P[7, 7] = 50.0

        # 装甲板数量及 ID 管理初始化
        self.armor_count = self._get_armor_count()
        self.id_counts = [0] * self.armor_count
        self.id_lost = [0] * self.armor_count
        self.stable_id = 0
        self.match_id = 0

        # 如果是前哨站，可调整观测噪声（对应 C++ 注释部分）
        # if self.robot_type == RobotType.Outpost:
        #     self.R = np.diag([2.0, 2.0, 2.0, 0.1])

    def predict(self, dt):
        if dt <= 0:
            return

        # ---------- 角度 KF 预测 ----------
        F_angle = np.array([[1, dt], [0, 1]])
        Q_angle = self.angle_Q_cont * dt
        self.angle_kf.predict(F_angle, Q_angle)

        # ---------- 主模型 EKF 预测 ----------
        Q_step = self.Q_cont * dt

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

        # 注意：EKF.predict 参数顺序为 (F, Q, f_func, F_jacobian)
        self.ekf.predict(F=None, Q=Q_step, f_func=f_func, F_jacobian=F_jacob)

    def update(self, obs):
        if not self.is_init or obs.robot_type != self.robot_type:
            self.init_model(obs)
            return

        obs_pos = obs.rel_pos
        obs_yaw = obs.rel_rpy[2]

        # ---------- 角度连续性处理 ----------
        angle_diff = safe_angle_sub(obs_yaw, self.last_obs_yaw)
        interval = (2.0 * np.pi) / self.armor_count
        threshold = interval / 2.0
        if abs(angle_diff) > threshold:
            corr = interval if angle_diff > 0 else -interval
            self.cont_yaw -= corr
        self.cont_yaw += angle_diff
        self.last_obs_yaw = obs_yaw

        # ---------- 角度 KF 更新 ----------
        z_angle = np.array([self.cont_yaw])
        H_angle = np.array([[1, 0]])
        def z_sub_angle(z, zp):
            diff = z - zp
            diff[0] = limit_rad(diff[0])
            return diff
        self.angle_kf.update(z_angle, H_angle, self.angle_R, z_sub_func=z_sub_angle)
        omega_kf = self.angle_kf.x[1]
        P_omega_kf = self.angle_kf.P[1, 1]

        # ---------- 匹配装甲板 ID 并获取对齐后的 yaw ----------
        match_result = self._solve_match_id(obs_pos, obs_yaw, self.match_id)
        best_id = match_result[0]
        self.match_id = best_id
        aligned_yaw = match_result[1]

        # ---------- 主状态 EKF 更新 ----------
        z_meas = np.array([obs_pos[0], obs_pos[1], obs_pos[2], aligned_yaw])

        def h_func(x):
            xc, yc, zc = x[0], x[1], x[2]
            psi = x[6]
            r, dz, _, _ = self._get_armor_params(best_id)
            armor_yaw = psi + best_id * (2.0 * np.pi / self.armor_count)
            return np.array([
                xc + r * np.cos(armor_yaw),
                yc + r * np.sin(armor_yaw),
                zc + dz,
                armor_yaw
            ])

        def H_jacob(x):
            H = np.zeros((4, self.state_dim))
            psi = x[6]
            r, dz, r_idx, dz_idx = self._get_armor_params(best_id)
            armor_yaw = psi + best_id * (2.0 * np.pi / self.armor_count)
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
            z=z_meas, H=None, R=self.R,
            z_sub_func=self._meas_sub,
            h_func=h_func, H_jacob=H_jacob
        )

        # ---------- 半径约束及特殊处理 ----------
        if self.robot_type == RobotType.Outpost:
            # ---------- 将角速度估计直接赋给主状态，并重置协方差 ----------
            self.ekf.x[7] = omega_kf
            print(omega_kf)
            # 交叉协方差清零
            self.ekf.P[:, 7] = 0.0
            self.ekf.P[7, :] = 0.0
            self.ekf.P[7, 7] = P_omega_kf

            R_MIN = 0.265
            R_MAX = 0.275
            if self.ekf.x[8] < R_MIN:
                self.ekf.x[8] = R_MIN
            elif self.ekf.x[8] > R_MAX:
                self.ekf.x[8] = R_MAX
        else:
            MIN_RADIUS = 0.1
            MAX_RADIUS = 0.40
            if self.ekf.x[8] < MIN_RADIUS:
                self.ekf.x[8] = MIN_RADIUS
            elif self.ekf.x[8] > MAX_RADIUS:
                self.ekf.x[8] = MAX_RADIUS
            if self.ekf.x[9] < MIN_RADIUS:
                self.ekf.x[9] = MIN_RADIUS
            elif self.ekf.x[9] > MAX_RADIUS:
                self.ekf.x[9] = MAX_RADIUS

    def _get_armor_count(self):
        """对应 AntitopV5::getArmorCount"""
        if self.robot_type == RobotType.Outpost:
            return 3
        return 4

    def _get_armor_params(self, k):
        """对应 AntitopV5::getArmorParams"""
        x = self.ekf.x
        if self.robot_type == RobotType.Outpost:
            if k == 0:
                return x[8], 0.0, 8, None
            elif k == 1:
                return x[8], x[10], 8, 10
            else:
                return x[8], x[11], 8, 11
        else:
            if k % 2 == 0:
                return x[8], 0.0, 8, None
            else:
                return x[9], x[10], 9, 10

    def _solve_match_id(self, obs_pos, obs_yaw, last_id):
        """完全对齐 AntitopV5::solveMatchID 逻辑"""
        center_yaw = self.ekf.x[6]
        armor_count = self.armor_count

        # 1. 计算每个装甲板预测角度与观测的角度差
        angle_diffs = []
        for i in range(armor_count):
            pred_armor_yaw = center_yaw + i * (2.0 * np.pi / armor_count)
            diff = abs(safe_angle_sub(obs_yaw, pred_armor_yaw))
            angle_diffs.append(diff)

        # 2. 找最小角度差的 ID
        best_id = int(np.argmin(angle_diffs))

        # 3. 强滞环逻辑：尽量保持上一匹配 ID
        if last_id >= 0 and last_id < armor_count:
            if angle_diffs[last_id] < self.keep_threshold or \
               (angle_diffs[last_id] < angle_diffs[best_id] + self.hysteresis_bias):
                best_id = last_id

        # 4. 计算对齐后的 yaw（保留残差）
        pred_best_yaw = center_yaw + best_id * (2.0 * np.pi / armor_count)
        yaw_residual = safe_angle_sub(obs_yaw, pred_best_yaw)
        aligned_yaw = pred_best_yaw + yaw_residual

        return best_id, aligned_yaw

    # ---------- 外部接口 ----------
    def get_all_armor_positions_at_time(self, dt):
        if not self.is_init:
            return [np.zeros(3)] * self.armor_count
        x = self.ekf.x
        pred_cx = x[0] + x[3] * dt
        pred_cy = x[1] + x[4] * dt
        pred_cz = x[2] + x[5] * dt
        pred_psi = x[6] + x[7] * dt
        armors = []
        for k in range(self.armor_count):
            r, dz, _, _ = self._get_armor_params(k)
            armor_yaw = pred_psi + k * (2.0 * np.pi / self.armor_count)
            ax = pred_cx + r * np.cos(armor_yaw)
            ay = pred_cy + r * np.sin(armor_yaw)
            az = pred_cz + dz
            armors.append(np.array([ax, ay, az]))
        return armors

    def get_geometric_distance(self, obs_pos):
        """用于数据关联的几何距离"""
        xc, yc, zc = self.ekf.x[0], self.ekf.x[1], self.ekf.x[2]
        psi = self.ekf.x[6]
        min_dist = float('inf')
        for k in range(self.armor_count):
            r, dz, _, _ = self._get_armor_params(k)
            armor_yaw = psi + k * (2.0 * np.pi / self.armor_count)
            pred_ax = xc + r * np.cos(armor_yaw)
            pred_ay = yc + r * np.sin(armor_yaw)
            pred_az = zc + dz
            dist = np.linalg.norm([obs_pos[0] - pred_ax, obs_pos[1] - pred_ay, obs_pos[2] - pred_az])
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def diverged(self):
        if not self.is_init:
            return True
        x = self.ekf.x
        if x[8] > 0.6 or x[9] > 0.6:
            return True
        if abs(x[3]) > 10 or abs(x[4]) > 10:
            return True
        return False
