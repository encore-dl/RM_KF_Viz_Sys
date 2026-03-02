import numpy as np


class QuinticTrajectory:
    """
    五次多项式轨迹：给定起点状态 (p0, v0, a0) 和终点状态 (p1, v1, a1) 以及总时间 T，
    生成平滑的位置、速度、加速度曲线。
    """
    def __init__(self, p0, v0, a0, p1, v1, a1, T):
        """
        参数：
            p0, v0, a0: 起点位置、速度、加速度
            p1, v1, a1: 终点位置、速度、加速度
            T: 总时间（>0）
        """
        self.T = T
        # 计算系数
        self.coeffs = self._compute_coeffs(p0, v0, a0, p1, v1, a1, T)

    @staticmethod
    def _compute_coeffs(p0, v0, a0, p1, v1, a1, T):
        """解五次多项式系数 a0~a5，其中位置 p(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5"""
        # 构建方程组矩阵 (6x6)
        A = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0],
            [1, T, T**2, T**3, T**4, T**5],
            [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],
            [0, 0, 2, 6*T, 12*T**2, 20*T**3]
        ])
        b = np.array([p0, v0, a0, p1, v1, a1])
        coeffs = np.linalg.solve(A, b)
        return coeffs

    def evaluate(self, t):
        """
        计算时刻 t (0 <= t <= T) 的位置、速度、加速度
        返回 (pos, vel, acc)
        """
        if t < 0:
            t = 0
        elif t > self.T:
            t = self.T
        # 位置
        pos = self.coeffs[0] + self.coeffs[1]*t + self.coeffs[2]*t**2 + \
              self.coeffs[3]*t**3 + self.coeffs[4]*t**4 + self.coeffs[5]*t**5
        # 速度
        vel = self.coeffs[1] + 2*self.coeffs[2]*t + 3*self.coeffs[3]*t**2 + \
              4*self.coeffs[4]*t**3 + 5*self.coeffs[5]*t**4
        # 加速度
        acc = 2*self.coeffs[2] + 6*self.coeffs[3]*t + 12*self.coeffs[4]*t**2 + 20*self.coeffs[5]*t**3
        return pos, vel, acc