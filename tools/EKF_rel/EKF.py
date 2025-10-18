import numpy as np
from collections import deque


class ExtendedKalmanFilter:
    """扩展卡尔曼滤波器 - 严格遵循源码接口"""

    def __init__(self, x0=None, P0=None, x_add_func=None):
        if x0 is None:
            # 默认状态: [x, vx, y, vy, z, vz, a, w, r, l, h]
            self.x = np.zeros(11)
        else:
            self.x = x0.copy()

        if P0 is None:
            self.P = np.eye(11)
        else:
            self.P = P0.copy()

        self.x_add_func = x_add_func if x_add_func else lambda a, b: a + b

        # 性能监控数据
        self.recent_nis_failures = deque(maxlen=10)
        self.window_size = 10
        self.data = {
            "residual_yaw": 0,
            "residual_pitch": 0,
            "residual_distance": 0,
            "residual_angle": 0,
            "nis": 0,
            "nees": 0,
            "nis_fail": False,
            "nees_fail": False,
            "recent_nis_failures": 0
        }

    def predict(self, F, Q, f_func=None):
        """预测步骤"""
        if f_func:
            self.x = f_func(self.x)
        else:
            self.x = F @ self.x

        self.P = F @ self.P @ F.T + Q

    def update(self, z, H, R, h_func, z_subtract_func):
        """更新步骤"""
        # 预测观测值
        z_pred = h_func(self.x)

        # 计算残差
        y = z_subtract_func(z, z_pred)

        # 计算卡尔曼增益
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        # 更新状态
        self.x = self.x_add_func(self.x, K @ y)
        self.P = (np.eye(11) - K @ H) @ self.P

        # NIS计算
        nis = y.T @ np.linalg.inv(S) @ y
        nis_threshold = 9.488  # 卡方检验，自由度=4，置信度95%
        nis_fail = nis > nis_threshold

        # 更新性能数据
        self.recent_nis_failures.append(nis_fail)
        self.data.update({
            "residual_yaw": y[0],
            "residual_pitch": y[1],
            "residual_distance": y[2],
            "residual_angle": y[3],
            "nis": nis,
            "nis_fail": nis_fail,
            "recent_nis_failures": sum(self.recent_nis_failures)
        })