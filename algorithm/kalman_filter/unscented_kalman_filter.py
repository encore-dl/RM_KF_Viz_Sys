import numpy as np
import scipy.linalg as spla


class UnscentedKalmanFilter:
    def __init__(self,
                 state_dim=None,
                 x0=None,
                 P0=None,
                 alpha=0.001,
                 beta=2.0,
                 kappa=0.0,
                 x_add_func=None,
                 x_sub_func=None,
                 x_mean_func=None,
                 z_sub_func=None,
                 z_mean_func=None
                 ):
        self.state_dim = None

        if state_dim is not None:
            self.state_dim = state_dim
            self.x = np.zeros(self.state_dim)

        # 从x0或P0推导状态维度
        if x0 is not None:
            self.state_dim = len(x0)
            self.x = x0.copy().flatten()
        elif P0 is not None:
            self.state_dim = P0.shape[0]
            self.x = np.zeros(self.state_dim)

        if self.state_dim is None:
            print("UnscentedKalmanFilter: state_dim error")
            raise ValueError

        # 初始化协方差矩阵
        if P0 is not None:
            self.P = P0.copy()
            # 验证维度一致性
            if self.P.shape != (self.state_dim, self.state_dim):
                raise ValueError("UnscentedKalmanFilter: P0 MISMATCH x0")
        else:
            self.P = np.eye(self.state_dim)

        # 验证x0和P0的维度一致性（如果两者都提供）
        if x0 is not None and P0 is not None:
            if len(x0) != P0.shape[0]:
                raise ValueError("UnscentedKalmanFilter: x0 MISMATCH P0")

        self.x_add_func = x_add_func if x_add_func else np.add
        self.x_sub_func = x_sub_func if x_sub_func else np.subtract
        self.x_mean_func = x_mean_func if x_mean_func else lambda a, b: np.dot(b, a)
        self.z_sub_func = z_sub_func if z_sub_func else np.subtract
        self.z_mean_func = z_mean_func if z_mean_func else lambda a, b: np.dot(b, a)

        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lambda_plus_n = 0.

        self.mean_weights = None
        self.cov_weights = None

        self._get_weights()

    def restart(self):
        self.x = np.zeros(self.state_dim)
        self.P = np.eye(self.state_dim)

    def _get_weights(self):
        n = self.state_dim
        lambda_ = self.alpha ** 2 * (n + self.kappa) - n

        # 均值权重
        self.mean_weights = np.full(2 * n + 1, 0.5 / (n + lambda_))
        self.mean_weights[0] = lambda_ / (n + lambda_)

        # 协方差权重
        self.cov_weights = np.full(2 * n + 1, 0.5 / (n + lambda_))
        self.cov_weights[0] = lambda_ / (n + lambda_) + (1 - self.alpha ** 2 + self.beta)

        self.lambda_plus_n = lambda_ + n

    def _get_sigma_points(self, x, P):
        n = self.state_dim
        sigma_points = np.zeros((2*n+1, n))

        P = (P + P.T) / 2.
        try:
            U = spla.cholesky(self.lambda_plus_n * P)
        except np.linalg.LinAlgError:
            print('UnscentedKalmanFilter: Cholesky failed, using SVD')
            u, s, vh = np.linalg.svd(P)
            U = (np.sqrt(s * self.lambda_plus_n) * u).T

        sigma_points[0] = x
        for k in range(n):
            sigma_points[k+1] = self.x_add_func(x, U[k])
            sigma_points[n+k+1] = self.x_sub_func(x, U[k])

        return sigma_points

    def _unscented_transform(self, sigma_points, mean_weights, cov_weights, cov_noise, mean_func, sub_func):
        mean = mean_func(sigma_points, mean_weights)

        n = len(mean)
        cov = np.zeros((n, n))
        for k in range(len(sigma_points)):
            diff = sub_func(sigma_points[k], mean)
            cov += cov_weights[k] * np.outer(diff, diff)
        cov += cov_noise

        return mean, cov

    def predict(self, Q, f_func):
        sigma_points = self._get_sigma_points(self.x, self.P)

        sigma_points_f = np.array([f_func(s) for s in sigma_points])

        self.x, self.P = self._unscented_transform(
            sigma_points_f,
            self.mean_weights,
            self.cov_weights,
            Q,
            self.x_mean_func,
            self.x_sub_func
        )

    def update(self, z, R, h_func, x_true=None):
        # 1. 将预测后的 Sigma 点映射到观测空间
        # 注意：这里使用的是 Predict 阶段产生的 sigmas_f
        sigma_points = self._get_sigma_points(self.x, self.P)
        sigma_points_h = np.array([h_func(s) for s in sigma_points])

        # 2. 计算观测的均值和协方差
        zp, S = self._unscented_transform(
            sigma_points_h,
            self.mean_weights,
            self.cov_weights,
            R,
            self.z_mean_func,
            self.z_sub_func
        )

        # 3. 计算状态与观测的互协方差矩阵 T
        T = np.zeros((self.state_dim, len(z)))
        for i in range(2 * self.state_dim + 1):
            diff_x = self.x_sub_func(sigma_points[i], self.x)
            diff_z = self.z_sub_func(sigma_points_h[i], zp)

            T += self.cov_weights[i] * np.outer(diff_x, diff_z)

        # 4. 卡尔曼增益
        # K = T @ S^-1
        K = T @ np.linalg.pinv(S)

        y = self.z_sub_func(z, zp)

        self.x = self.x_add_func(self.x, K @ y)
        self.P = self.P - K @ T.T

        self.P = (self.P + self.P.T) / 2.

        # 计算nis和nees等，暂定












