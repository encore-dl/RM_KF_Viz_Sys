import numpy as np

from algorithm.kalman_filter.extended_kalman_filter import ExtendedKalmanFilter
from object.model.demo2.data_and_utils.math_tools import (
    limit_rad, safe_angle_sub
)


class DemoModel2:
    def __init__(self):
        # [x, y, z, vx, vy, vz, ψ, ω, ra, rb, Δz]
        self.state_dim = 11

        # 初始化 EKF
        # 注意：这里 x_add_func 使用普通加法，允许 psi 连续增长 (例如 100pi)
        # 这样可以避免相位解缠时的跳变问题
        self.ekf = ExtendedKalmanFilter(
            state_dim=self.state_dim,
            x_add_func=lambda x, d: x + d
        )

        # P 矩阵 (初始协方差)
        self.ekf.P = np.eye(self.state_dim)
        self.ekf.P[6, 6] = 1.0  # psi(yaw)
        self.ekf.P[7, 7] = 200.0  # w (初始对转速很不确定)
        self.ekf.P[8, 8] = 0.01  # ra
        self.ekf.P[9, 9] = 0.01  # rb

        q_x = 0.01
        q_y = 0.01
        q_z = 0.01
        q_vx = 5.0  # 给大一点，允许速度快速变化，从而解释位置误差
        q_vy = 5.0
        q_vz = 5.0
        q_yaw = 1e-4
        q_omg = 0.5  # 允许角速度变化 (起转/停转)
        q_ra = 1e-7  # 结构参数极小，视为刚体，几乎不许变
        q_rb = 1e-7
        q_dz = 1e-7
        self.Q = np.diag([
            q_x, q_y, q_z,
            q_vx, q_vy, q_vz,
            q_yaw, q_omg,
            q_ra, q_rb, q_dz
        ])

        # R 矩阵 (观测噪声)
        r_x = 0.05
        r_y = 0.05
        r_z = 0.05
        r_yaw = 0.02
        self.R = np.diag([r_x, r_y, r_z, r_yaw])

        self.last_t = 0
        self.is_init = False
        self.match_id = 0
        self.default_r = 0.2
        self.spin_thresh = 2.0

    @staticmethod
    def _meas_sub(z, z_pred):
        """观测残差计算"""
        diff = z - z_pred
        # 只有 yaw (index 3) 需要处理角度差
        # 虽然我们尽量让 yaw 连续，但防止浮点误差，这里加个 safe sub
        diff[3] = limit_rad(diff[3])
        return diff

    def init_model(self, obs_pos, obs_yaw, t):
        x0 = np.zeros(self.state_dim)

        # 简单反解初始位置
        x0[0] = obs_pos[0] - self.default_r * np.cos(obs_yaw)
        x0[1] = obs_pos[1] - self.default_r * np.sin(obs_yaw)
        x0[2] = obs_pos[2]

        x0[6] = obs_yaw  # psi
        x0[8] = self.default_r  # ra
        x0[9] = self.default_r  # rb

        self.ekf.x = x0
        # 重置 P，确保初始化时有足够的收敛空间
        self.ekf.P = np.eye(self.state_dim) * 0.1
        self.ekf.P[7, 7] = 50.0

        self.last_t = t
        self.is_init = True

    def predict(self, dt):
        if dt <= 0: return

        # 1. 状态转移函数 f(x)
        # CV (匀速) + CT (匀转速) 模型
        # 舍弃了加速度状态，因为加速度噪声太大，容易引入过冲
        def f_func(x):
            nx = x.copy()
            # Pos += Vel * dt
            nx[0] += x[3] * dt
            nx[1] += x[4] * dt
            nx[2] += x[5] * dt
            # Psi += w * dt
            nx[6] += x[7] * dt
            return nx

        # 2. 状态转移雅可比 F
        # 这是一个稀疏矩阵，大部分是 Identity
        def F_jacob(x):
            F = np.eye(self.state_dim)
            F[0, 3] = dt
            F[1, 4] = dt
            F[2, 5] = dt
            F[6, 7] = dt
            return F

        # 过程噪声随 dt 缩放
        Q_step = self.Q * dt

        self.ekf.predict(F=None, Q=Q_step, f_func=f_func, F_jacobian=F_jacob)

    def update(self, obs_pos, obs_yaw, t):
        if not self.is_init:
            self.init_model(obs_pos, obs_yaw, t)
            return

        dt = t - self.last_t
        self.last_t = t

        # 1. Predict
        self.predict(dt)

        # 2. Match ID (相位对齐)
        # 获取预测的主轴角度
        pred_psi = self.ekf.x[6]
        # 计算观测到的 yaw 对应哪块板，并计算出“对齐后”的 yaw (aligned_yaw)
        # aligned_yaw 是连续的，解决了 -pi/pi 跳变问题
        self.match_id, aligned_yaw = self._solve_match_id(obs_yaw, pred_psi)

        # 3. Update
        # 构造观测向量 z: [x_obs, y_obs, z_obs, aligned_yaw]
        z_meas = np.array([obs_pos[0], obs_pos[1], obs_pos[2], aligned_yaw])

        # 确定当前使用的结构参数
        is_even = (self.match_id % 2 == 0)

        # --- 定义观测函数 h(x) ---
        def h_func(x):
            xc, yc, zc = x[0], x[1], x[2]
            psi = x[6]
            ra, rb, dz = x[8], x[9], x[10]

            r = ra if is_even else rb
            z_offset = 0.0 if is_even else dz

            # 当前观测板子的角度 = 主轴角度 + ID偏移
            armor_yaw = psi + self.match_id * (np.pi / 2.0)

            return np.array([
                xc + r * np.cos(armor_yaw),
                yc + r * np.sin(armor_yaw),
                zc + z_offset,
                armor_yaw
            ])

        # --- 定义观测雅可比 H(x) ---
        # 这是解决耦合问题的核心，它告诉滤波器位置和半径如何相互影响
        def H_jacob(x):
            H = np.zeros((4, self.state_dim))
            psi = x[6]
            ra, rb = x[8], x[9]

            r = ra if is_even else rb
            armor_yaw = psi + self.match_id * (np.pi / 2.0)

            c = np.cos(armor_yaw)
            s = np.sin(armor_yaw)

            # Row 0: x_obs
            H[0, 0] = 1.0  # dx/dxc
            H[0, 6] = -r * s  # dx/dpsi (链式: -r*sin * 1)
            if is_even:
                H[0, 8] = c  # dx/dra
            else:
                H[0, 9] = c  # dx/drb

            # Row 1: y_obs
            H[1, 1] = 1.0  # dy/dyc
            H[1, 6] = r * c  # dy/dpsi
            if is_even:
                H[1, 8] = s
            else:
                H[1, 9] = s

            # Row 2: z_obs
            H[2, 2] = 1.0
            if not is_even:
                H[2, 10] = 1.0  # dz/ddz

            # Row 3: yaw_obs
            H[3, 6] = 1.0  # dyaw/dpsi

            return H

        self.ekf.update(
            z=z_meas,
            H=None,
            R=self.R,
            z_sub_func=self._meas_sub,
            h_func=h_func,
            H_jacob=H_jacob
        )

    @staticmethod
    def _solve_match_id(obs_yaw, pred_psi):
        best_id = 0
        min_diff = float('inf')
        aligned_yaw = obs_yaw  # Default fallback

        # 寻找最接近观测角度的板子ID
        # 我们需要找到一个 k，使得 (pred_psi + k*pi/2) 与 obs_yaw 差距最小
        for i in range(4):
            # 假设当前是第 i 号板，反推对应的主轴角度 raw_psi
            # raw_psi = obs_yaw - i * pi/2
            # 看看这个 raw_psi 和 pred_psi 差多少 (考虑周期性)

            # 这种写法更直观：计算观测yaw转到标准位置后，与预测psi的差
            raw_diff = (obs_yaw - i * (np.pi / 2.0)) - pred_psi
            diff = limit_rad(raw_diff)

            if abs(diff) < min_diff:
                min_diff = abs(diff)
                best_id = i
                # 对齐后的yaw：预测值 + 偏差，保持连续性
                aligned_yaw = pred_psi + diff + i * (np.pi / 2.0)

        return best_id, aligned_yaw

    def get_pred_pos(self, fly_t):
        if not self.is_init:
            return np.zeros(3), np.zeros(3)

        x = self.ekf.x
        dt = fly_t

        # 1. 预测中心 (CV模型外推)
        pred_cx = x[0] + x[3] * dt
        pred_cy = x[1] + x[4] * dt
        pred_cz = x[2] + x[5] * dt

        # 2. 预测角度
        psi_pred = x[6] + x[7] * dt

        # 3. 确定击打策略
        is_even = (self.match_id % 2 == 0)
        r = x[8] if is_even else x[9]
        dz = 0.0 if is_even else x[10]

        # 策略判断：
        # 如果转速低 (Spinning off)，跟随当前板子
        # 如果转速高 (Spinning on)，击打最优点(面向枪口)
        if abs(x[7]) < self.spin_thresh:
            # 随动模式：预测 t 时刻当前板子的位置
            armor_yaw = psi_pred + self.match_id * (np.pi / 2.0)
        else:
            # 锁定模式：击打 t 时刻面向我的位置
            # 面向我的角度 = atan2(y_robot - y_gun, x_robot - x_gun) + pi
            # 假设枪口在原点 (0,0)
            yaw_to_self = np.atan2(-pred_cy, -pred_cx)
            armor_yaw = yaw_to_self

        pred_armor = [
            pred_cx + r * np.cos(armor_yaw),
            pred_cy + r * np.sin(armor_yaw),
            pred_cz + dz
        ]

        return [pred_cx, pred_cy, pred_cz], pred_armor


"""
    [Mathematical Model: Unified 11-DOF EKF]

    1. State Vector X (11x1):
       X = [xc, yc, zc, vx, vy, vz, psi, w, ra, rb, dz]^T
       ------------------------------------------------------
       xc, yc, zc : Robot Center Position (World)
       vx, vy, vz : Center Velocity
       psi        : Yaw angle of Armor #0 (Continuous)
       w          : Angular Velocity
       ra         : Radius for Armors 0 & 2 (Even)
       rb         : Radius for Armors 1 & 3 (Odd)
       dz         : Height offset for Odd armors relative to Even

    2. Process Model (Prediction):
       Motion: Constant Velocity (CV) + Constant Turn Rate (CT)
       X_{k+1} = F @ X_k
       ------------------------------------------------------
       xc_{k+1}  = xc_k + vx_k * dt
       yc_{k+1}  = yc_k + vy_k * dt
       zc_{k+1}  = zc_k + vz_k * dt
       psi_{k+1} = psi_k + w_k * dt
       * Others remain constant (Identity mapping)

    3. Measurement Model (Observation):
       Given matched armor ID 'k' (0-3):
       Current Angle: theta = psi + k * (pi/2)
       Selection:     r = ra if even else rb
                      dh = 0  if even else dz

       Z_pred = h(X) = [
           xc + r * cos(theta),   # x_obs
           yc + r * sin(theta),   # y_obs
           zc + dh,               # z_obs
           theta                  # yaw_obs
       ]

    4. Jacobian H = dh/dX (4x11):
       Let S = sin(theta), C = cos(theta)

       IDX:   0 1 2  3 4 5    6     7    8(ra)       9(rb)       10(dz)
       ------------------------------------------------------------------
       x_obs: 1 0 0  0 0 0  -r*S    0    C(if ev)    C(if od)    0
       y_obs: 0 1 0  0 0 0   r*C    0    S(if ev)    S(if od)    0
       z_obs: 0 0 1  0 0 0    0     0    0           0           1(if od)
       yaw:   0 0 0  0 0 0    1     0    0           0           0

       * Note: H[0,6] and H[1,6] couple Position with Rotation.
       * Note: H[0,8/9] couples Position with Radius.
    """