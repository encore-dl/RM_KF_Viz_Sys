import numpy as np
from collections import deque


# 状态向量默认状态: [x, vx, y, vy, z, vz, a, w, r, l, h] 11维向量
class ExtendedKalmanFilter:
    def __init__(self, x0=None, P0=None, x_add_func=None):
        # 从x0或P0推导状态维度
        if x0 is not None:
            self.state_dim = len(x0)
            self.x = x0.copy().flatten()
        elif P0 is not None:
            self.state_dim = P0.shape[0]
            self.x = np.zeros(self.state_dim)
        else:
            # 默认情况
            self.state_dim = 11
            self.x = np.zeros(self.state_dim)

        # 初始化协方差矩阵
        if P0 is not None:
            self.P = P0.copy()
            # 验证维度一致性
            if self.P.shape != (self.state_dim, self.state_dim):
                raise ValueError("P0 MISMATCH x0")
        else:
            self.P = np.eye(self.state_dim)

        # 验证x0和P0的维度一致性（如果两者都提供）
        if x0 is not None and P0 is not None:
            if len(x0) != P0.shape[0]:
                raise ValueError("x0 MISMATCH P0")

        self.x_add_func = x_add_func if x_add_func else lambda a, b: a + b

        # 性能监控数据
        self.performance_window = 50
        self.recent_nis_failures = deque(maxlen=self.performance_window)
        self.recent_nees_values = deque(maxlen=self.performance_window)
        self.recent_nees_failures = deque(maxlen=self.performance_window)

        self.data = {
            "residual_yaw": 0,
            "residual_pitch": 0,
            "residual_distance": 0,
            "residual_angle": 0,
            "nis": 0,
            "nees": 0,
            "nis_fail": False,
            "nees_fail": False,
            "recent_nis_failures": 0,
            "recent_nees_failures": 0,
            "nis_pass_rate": 0.0,
            "nees_pass_rate": 0.0
        }

        # 统计阈值 (95% 置信度)
        self.nis_threshold = 9.488  # 卡方检验，自由度=4
        self.nees_threshold = 19.675  # 卡方检验，自由度=11 (状态维度)

    def predict(self, F, Q, f_func=None, F_jacobian=None):
        # KF的预测两公式
        # 预测状态
        if f_func is not None:  # 非线性
            self.x = f_func(self.x)
            if F_jacobian is not None:
                F = F_jacobian(self.x)
        else:
            self.x = F @ self.x  # 线性

        # 协方差的预测
        self.P = F @ self.P @ F.T + Q

    def update(self, z, H, R, h_func, z_subtract_func, x_true=None):
        # KF的更新三公式
        if h_func is not None:
            z_pred = h_func(self.x)
        else:
            z_pred = H @ self.x

        # 计算残差
        y = z_subtract_func(z, z_pred)

        # 计算卡尔曼增益
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        # 更新状态
        self.x = self.x_add_func(self.x, K @ y)
        # 更新协方差
        I = np.eye(self.state_dim)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T

        self._evaluate_performance(y, S, x_true)

    def _evaluate_performance(self, y, S, x_true=None):
        # NIS计算
        nis = y.T @ np.linalg.inv(S) @ y
        nis_fail = nis > self.nis_threshold

        self.recent_nis_failures.append(nis_fail)
        nis_failure_rate = sum(self.recent_nis_failures) / len(self.recent_nis_failures)

        # NEES计算 (如果有真实状态)
        nees = 0
        nees_fail = False
        if x_true is not None:
            error = x_true - self.x
            nees = error.T @ np.linalg.inv(self.P) @ error
            nees_fail = nees > self.nees_threshold

            self.recent_nees_values.append(nees)
            self.recent_nees_failures.append(nees_fail)
            nees_failure_rate = sum(self.recent_nees_failures) / len(self.recent_nees_failures)
        else:
            nees_failure_rate = 0.0

        # 更新性能数据
        self.data.update({
            "residual_yaw": y[0] if len(y) > 0 else 0,
            "residual_pitch": y[1] if len(y) > 1 else 0,
            "residual_distance": y[2] if len(y) > 2 else 0,
            "residual_angle": y[3] if len(y) > 3 else 0,
            "nis": nis,
            "nees": nees,
            "nis_fail": nis_fail,
            "nees_fail": nees_fail,
            "recent_nis_failures": sum(self.recent_nis_failures),
            "recent_nees_failures": sum(self.recent_nees_failures),
            "nis_pass_rate": 1.0 - nis_failure_rate,
            "nees_pass_rate": 1.0 - nees_failure_rate if x_true is not None else 0.0
        })
