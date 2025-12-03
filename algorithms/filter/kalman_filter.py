import numpy as np


class KalmanFilter:
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
                raise ValueError("P0 MISMATCH x0!")
        else:
            self.P = np.eye(self.state_dim)

        # 验证x0和P0的维度一致性（如果两者都提供）
        if x0 is not None and P0 is not None:
            if len(x0) != P0.shape[0]:
                raise ValueError("x0 MISMATCH P0!")

        self.x_add_func = x_add_func if x_add_func else lambda a, b: a + b

    def predict(self, F, Q):
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z, H, R):
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ (z - H @ self.x)
        I = np.eye(self.state_dim)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T


