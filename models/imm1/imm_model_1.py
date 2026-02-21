import time
import numpy as np

from core.algorithms.filters.kalman import KalmanFilter
from core.algorithms.filters.unscented_kalman import UnscentedKalmanFilter
from models.imm1.utils.math import (
    limit_rad
)

# 状态索引映射 (14维)
# Motion (0-7): x, y, z, vx, vy, vz, ax, ay
# Rot (8-13):   psi, w, alpha, ra, rb, dz
IDX_X, IDX_Y, IDX_Z = 0, 1, 2
IDX_VX, IDX_VY, IDX_VZ = 3, 4, 5
IDX_AX, IDX_AY = 6, 7
IDX_PSI, IDX_W, IDX_ALP = 8, 9, 10
IDX_RA, IDX_RB, IDX_DZ = 11, 12, 13


class IMMModel1:
    def __init__(self):
        # --- IMM 参数 ---
        self.state_dim = 14
        self.n_models = 2  # 0:KF(Motion), 1:UKF(Spin)

        # 转移概率矩阵
        self.trans_matrix = np.array([
            [0.85, 0.15],  # KF -> UKF 的意愿变强
            [0.15, 0.85]  # UKF -> KF 的意愿变强
        ])
        # 模型概率
        self.mu = np.array([0.5, 0.5])

        # 融合后的状态
        self.fused_x = np.zeros(self.state_dim)
        self.fused_P = np.eye(self.state_dim)

        # --- 原始参数 ---
        self.default_r = 0.2
        self.F_a_dec = 0.6
        self.spin_thresh = 2.
        self.match_id = 0
        self.last_t = time.time()
        self.is_init = False
        self.c_bar = np.zeros(2)  # 归一化常数缓存

        # --- 初始化滤波器 ---

        # 1. KF (Motion Dominant)
        # 即使是 KF，我们也扩维到 14 维，方便 IMM 混合
        self.kf_motion = KalmanFilter(state_dim=self.state_dim)

        # 2. UKF (Spin Dominant)
        # 这里的 z_sub_func 会在 UKF 内部 update 计算 y = z - zp 时调用
        # 必须传入处理角度的减法函数
        self.ukf_rot = UnscentedKalmanFilter(
            state_dim=self.state_dim,
            alpha=0.1, beta=2., kappa=0.,
            x_add_func=self._state_add,  # 状态叠加：Yaw 不解缠绕，保持连续
            x_sub_func=self._state_sub,  # 状态相减
            z_sub_func=self._z_sub_obs  # 观测相减：Yaw 需要解缠绕 (limit_rad)
        )

        self._init_matrices()

    def _init_matrices(self):
        # 初始化 P (协方差)
        P_motion = np.eye(8) * 1.0
        P_rot = np.diag([1.0, 1.0, 1.0, 100.0, 100.0, 1.0])
        P_full = np.zeros((14, 14))
        P_full[:8, :8] = P_motion
        P_full[8:, 8:] = P_rot

        self.kf_motion.P = P_full.copy()
        self.ukf_rot.P = P_full.copy()

        # 初始化 Q (过程噪声)
        # KF: 信任平移，旋转部分给大噪声(由观测修正)
        self.Q_kf = np.eye(14) * 1e-4
        Q_motion_part = np.diag([
            0.05, 0.05, 0.001,  # pos
            0.20, 0.20, 0.20,  # vel
            50., 50.  # acc
        ])
        self.Q_kf[:8, :8] = Q_motion_part
        self.Q_kf[8:, 8:] *= 10.0

        # UKF: 信任旋转模型
        self.Q_ukf = np.eye(14) * 1e-4
        Q_rot_part = np.diag([
            1e-4, 0.05, 1.0,  # psi, w, alpha
            0.01, 0.01, 1e-5  # ra, rb, dz
        ])
        self.Q_ukf[:8, :8] = Q_motion_part
        self.Q_ukf[8:, 8:] = Q_rot_part

        # 初始化 R (观测噪声)
        # 统一观测向量 z = [x, y, z, yaw]
        self.R_common = np.diag([0.01, 0.01, 0.01, 0.02])

    # --- 核心辅助函数 ---

    @staticmethod
    def _state_add(x1, x2):
        # 状态更新/Sigma点恢复：Yaw 保持连续性，绝不使用 limit_rad
        return x1 + x2

    @staticmethod
    def _state_sub(x1, x2):
        # Sigma点生成：普通减法
        return x1 - x2

    @staticmethod
    def _z_sub_obs(z1, z2):
        # 观测空间残差计算：Yaw 必须解缠绕到 [-pi, pi]
        # 假设 z = [x, y, z, yaw]
        diff = z1 - z2
        if len(diff) > 3:
            diff[3] = limit_rad(diff[3])
        return diff

    def init_model(self, obs_pos, obs_yaw, t):
        x_init = np.zeros(self.state_dim)

        # 初始假设：观测点即为装甲板，逆推中心需要半径
        # 刚开始不知道 match_id，假设它是 0 号板
        yaw_init = obs_yaw

        x_init[IDX_X] = obs_pos[0] - self.default_r * np.cos(yaw_init)
        x_init[IDX_Y] = obs_pos[1] - self.default_r * np.sin(yaw_init)
        x_init[IDX_Z] = obs_pos[2]

        x_init[IDX_PSI] = yaw_init
        x_init[IDX_RA] = self.default_r
        x_init[IDX_RB] = self.default_r

        self.kf_motion.x = x_init.copy()
        self.ukf_rot.x = x_init.copy()

        # 重置 P
        self.kf_motion.P *= 10.
        self.ukf_rot.P *= 10.

        self.fused_x = x_init.copy()
        self.mu = np.array([0.5, 0.5])
        self.last_t = t
        self.is_init = True

    def _interact(self):
        # IMM 步骤1: 混合
        # 计算混合概率
        c = np.dot(self.mu, self.trans_matrix)
        mu_mix = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                mu_mix[i, j] = self.trans_matrix[i, j] * self.mu[i] / c[j]

        xs = [self.kf_motion.x, self.ukf_rot.x]
        Ps = [self.kf_motion.P, self.ukf_rot.P]

        x0_mix = []
        P0_mix = []

        for j in range(2):
            # 混合状态
            x_m = np.zeros(self.state_dim)
            for i in range(2):
                # 线性加权，Yaw 保持连续性
                x_m += mu_mix[i, j] * xs[i]
            x0_mix.append(x_m)

            # 混合协方差
            P_m = np.zeros((self.state_dim, self.state_dim))
            for i in range(2):
                diff = xs[i] - x_m
                # 混合时的协方差计算，Yaw 差异需要处理一下吗？
                # 如果两个模型的 Yaw 已经缠绕到相差 2pi 以上，直接减会很大
                # 但只要保持两个模型都是连续的，且初始一致，偏差应该不会超过 2pi
                # 为了保险，这里对 Yaw 做差可以使用 safe_sub (unwrap) 逻辑
                # 但 x_m 本身是加权平均值，这里直接减通常问题不大，除非 mu 极不平衡且 yaw 差一圈
                # 这里为了严谨，对 Yaw 维度做一次 limit_rad
                diff[IDX_PSI] = limit_rad(diff[IDX_PSI])

                P_m += mu_mix[i, j] * (Ps[i] + np.outer(diff, diff))
            P0_mix.append(P_m)

        return x0_mix, P0_mix, c

    def predict(self, dt):
        if dt <= 0: return

        # 1. 交互
        x0, P0, self.c_bar = self._interact()

        self.kf_motion.x = x0[0]
        self.kf_motion.P = P0[0]
        self.ukf_rot.x = x0[1]
        self.ukf_rot.P = P0[1]

        # 2. 预测
        # --- KF Predict ---
        F = np.eye(14)
        # Motion (CA)
        F[IDX_X, IDX_VX] = dt
        F[IDX_X, IDX_AX] = 0.5 * dt ** 2
        F[IDX_Y, IDX_VY] = dt
        F[IDX_Y, IDX_AY] = 0.5 * dt ** 2
        F[IDX_Z, IDX_VZ] = dt
        F[IDX_VX, IDX_AX] = dt
        F[IDX_VY, IDX_AY] = dt
        F[IDX_AX, IDX_AX] = self.F_a_dec
        F[IDX_AY, IDX_AY] = self.F_a_dec
        # Rot (CV for KF)
        F[IDX_PSI, IDX_W] = dt

        self.kf_motion.predict(F=F, Q=self.Q_kf * dt)

        # --- UKF Predict ---
        def f_func_ukf(x):
            next_x = x.copy()
            # Motion: CA
            next_x[IDX_X] += x[IDX_VX] * dt + 0.5 * x[IDX_AX] * dt ** 2
            next_x[IDX_Y] += x[IDX_VY] * dt + 0.5 * x[IDX_AY] * dt ** 2
            next_x[IDX_Z] += x[IDX_VZ] * dt
            next_x[IDX_VX] += x[IDX_AX] * dt
            next_x[IDX_VY] += x[IDX_AY] * dt
            next_x[IDX_AX] *= self.F_a_dec
            next_x[IDX_AY] *= self.F_a_dec

            # Rot: Spin Model
            # psi 累加，保持连续
            next_x[IDX_PSI] += x[IDX_W] * dt + 0.5 * x[IDX_ALP] * dt ** 2
            next_x[IDX_W] += x[IDX_ALP] * dt
            next_x[IDX_ALP] *= 0.95
            return next_x

        self.ukf_rot.predict(Q=self.Q_ukf * dt, f_func=f_func_ukf)

    def update(self, obs_pos, obs_yaw, t):
        if not self.is_init:
            self.init_model(obs_pos, obs_yaw, t)
            return

        dt = t - self.last_t
        self.last_t = t

        self.predict(dt)

        # 匹配 ID (使用 UKF 预测的 Yaw，因为它包含角加速度，通常更准)
        pred_yaw_ukf = self.ukf_rot.x[IDX_PSI]
        self.match_id, aligned_yaw = self._solve_match_id(obs_yaw, pred_yaw_ukf)

        # 统一观测向量 z = [x, y, z, aligned_yaw]
        z_meas = np.array([obs_pos[0], obs_pos[1], obs_pos[2], aligned_yaw])

        # --- KF Update ---
        # 构造线性观测矩阵 H
        # 我们需要 KF 也能观测到 "装甲板位置"，但 KF 状态存的是 "车中心"
        # 线性化： z_meas = H * x + noise
        # 这里的 H 对于 PSI 直接是 1
        # 对于 Pos，KF 预测的是 Center，观测的是 Armor。
        # 技巧：我们将观测值 z_meas 减去 "预测的半径向量"，伪装成对 Center 的观测
        # 或者：为了保持数学严谨性，我们让 KF 仍然观测 Center，但在 IMM 似然比对时需要一致。
        # 最佳方案：修改 z_meas 送给 KF 的值。

        # 计算当前使用的半径和 dz
        # 注意：这里使用 fused_x 或 KF 自己的 x 都可以，用 KF 自己的更符合卡尔曼假设
        kf_r = self.kf_motion.x[IDX_RA] if self.match_id % 2 == 0 else self.kf_motion.x[IDX_RB]
        kf_dz = 0.0 if self.match_id % 2 == 0 else self.kf_motion.x[IDX_DZ]

        # 预测的装甲板朝向 (基于 KF)
        kf_armor_yaw = self.kf_motion.x[IDX_PSI] + self.match_id * (np.pi / 2)

        # 将装甲板观测转换为车中心观测 (Inverse Observation)
        obs_center_x = obs_pos[0] - kf_r * np.cos(kf_armor_yaw)
        obs_center_y = obs_pos[1] - kf_r * np.sin(kf_armor_yaw)
        obs_center_z = obs_pos[2] - kf_dz

        z_for_kf = np.array([obs_center_x, obs_center_y, obs_center_z, aligned_yaw])

        H_kf = np.zeros((4, 14))
        H_kf[0, IDX_X] = 1
        H_kf[1, IDX_Y] = 1
        H_kf[2, IDX_Z] = 1
        H_kf[3, IDX_PSI] = 1

        # 传入 z_sub_func 处理 Yaw 环绕
        self.kf_motion.update(
            z=z_for_kf,
            H=H_kf,
            R=self.R_common,
            z_sub_func=self._z_sub_obs
        )

        # --- UKF Update ---
        # UKF 可以直接建立非线性观测模型 h(x) -> Armor Pos
        def h_func_ukf(x):
            is_even = (self.match_id % 2 == 0)
            r = x[IDX_RA] if is_even else x[IDX_RB]
            dz = 0. if is_even else x[IDX_DZ]

            # 状态里的 PSI 是车头朝向 (连续值)
            # 装甲板朝向 = 车头 + id * pi/2
            tar_yaw = x[IDX_PSI] + self.match_id * (np.pi / 2)

            px = x[IDX_X] + r * np.cos(tar_yaw)
            py = x[IDX_Y] + r * np.sin(tar_yaw)
            pz = x[IDX_Z] + dz

            # 返回 [x, y, z, yaw_aligned]
            # 注意：这里的 yaw 直接返回 tar_yaw (连续值)，z_sub_func 会负责减法和取余
            return np.array([px, py, pz, tar_yaw])

        # UKF 初始化时已经设定了 z_sub_func
        self.ukf_rot.update(
            z=z_meas,
            R=self.R_common,
            h_func=h_func_ukf
        )

        # --- IMM 步骤3: 概率更新 ---
        # 使用你将在 Filter 中暴露的 self.y (残差) 和 self.S (新息协方差)

        # KF 的残差是在 Center 空间的，UKF 是在 Armor 空间的
        # 但因为我们将 R 设为一样，且变换是刚性的，它们的 Likelihood 在数量级上是可比的
        like_kf = self._calc_likelihood(self.kf_motion.y, self.kf_motion.S)
        like_ukf = self._calc_likelihood(self.ukf_rot.y, self.ukf_rot.S)

        probs = np.array([like_kf, like_ukf])
        mu_raw = probs * self.c_bar

        sum_mu = np.sum(mu_raw)
        if sum_mu > 1e-12:
            self.mu = mu_raw / sum_mu
        else:
            self.mu = np.array([0.5, 0.5])

        # --- IMM 步骤4: 融合 ---
        self.fused_x = np.zeros(self.state_dim)
        for i, model in enumerate([self.kf_motion, self.ukf_rot]):
            self.fused_x += self.mu[i] * model.x  # 线性叠加，连续 Yaw

        self.fused_P = np.zeros((self.state_dim, self.state_dim))
        for i, model in enumerate([self.kf_motion, self.ukf_rot]):
            diff = model.x - self.fused_x
            # P 的融合需要处理 Yaw 的差值
            diff[IDX_PSI] = limit_rad(diff[IDX_PSI])
            self.fused_P += self.mu[i] * (model.P + np.outer(diff, diff))

    def _calc_likelihood(self, y, S):
        try:
            d = len(y)
            S_inv = np.linalg.inv(S)
            denom = np.sqrt((2 * np.pi) ** d * np.linalg.det(S))
            exponent = -0.5 * np.dot(y.T, np.dot(S_inv, y))
            return np.exp(exponent) / (denom + 1e-9)
        except:
            return 1e-9

    @staticmethod
    def _solve_match_id(obs_yaw, pred_yaw):
        # 这里的 pred_yaw 是连续值 (比如 100.0)
        # obs_yaw 是观测值 (比如 0.5)
        # 我们要找到一个 k，使得 obs_yaw + 2k*pi 与 pred_yaw + match_off 最近

        best_id = 0
        min_diff = float('inf')
        aligned_yaw = pred_yaw  # 默认为预测值

        # 假设 pred_yaw 已经包含了之前的圈数信息
        # 我们只需看 obs_yaw 在四种装甲板位置中，哪一种对应的 "解缠绕后角度" 离 pred_yaw 最近

        for i in range(4):
            # 假设当前观测的是第 i 号装甲板
            # 那么车头朝向应该是 obs_yaw - i * pi/2
            raw_yaw_diff = obs_yaw - i * (np.pi / 2.)

            # 计算观测与预测的偏差 (限制在 -pi, pi)
            diff = limit_rad(raw_yaw_diff - pred_yaw)

            # 还原到连续空间： predicted + diff
            # 这里的 aligned 候选值
            candidate_aligned = pred_yaw + diff

            if abs(diff) < min_diff:
                min_diff = abs(diff)
                best_id = i
                aligned_yaw = candidate_aligned

        return best_id, aligned_yaw

    def get_pred_pos(self, fly_t):
        x = self.fused_x
        dt = fly_t

        # 运动预测 CA
        pred_cx = x[IDX_X] + x[IDX_VX] * dt + 0.5 * x[IDX_AX] * (dt ** 2)
        pred_cy = x[IDX_Y] + x[IDX_VY] * dt + 0.5 * x[IDX_AY] * (dt ** 2)
        pred_cz = x[IDX_Z] + x[IDX_VZ] * dt

        pred_center = [pred_cx, pred_cy, pred_cz]

        # 旋转预测 (CV+Alpha)
        # Yaw 保持连续
        pred_yaw = x[IDX_PSI] + x[IDX_W] * dt + 0.5 * x[IDX_ALP] * (dt ** 2)

        tar_id = self.match_id
        is_even = (tar_id % 2 == 0)
        r_pred = x[IDX_RA] if is_even else x[IDX_RB]
        dz_pred = 0. if is_even else x[IDX_DZ]

        # 计算装甲板位置
        tar_yaw = pred_yaw + tar_id * (np.pi / 2.)

        pred_armor = [
            pred_center[0] + r_pred * np.cos(tar_yaw),
            pred_center[1] + r_pred * np.sin(tar_yaw),
            pred_center[2] + dz_pred
        ]

        return [pred_center, pred_armor]