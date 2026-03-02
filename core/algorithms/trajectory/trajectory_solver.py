import numpy as np

class TrajectorySolver:
    def __init__(self, g=9.8, k=0.001, N=100):
        self.g = g
        self.k = k
        self.N = N

    def solve(self, v0, target_pos, pitch_guess, max_iter=20, tol=1e-7):
        x0 = np.sqrt(target_pos[0]**2 + target_pos[1]**2)  # 水平距离
        y0 = target_pos[2]                                 # 竖直高度
        # 初始参数
        r1 = np.tan(pitch_guess)                           # 初始倾角正切
        # 粗略飞行时间估算（用于初始 r0）
        t_est = x0 / (v0 * np.cos(pitch_guess)) if abs(np.cos(pitch_guess)) > 1e-6 else 0.0
        r0 = (v0 * np.sin(pitch_guess) - self.g * t_est) / (v0 * np.cos(pitch_guess)) if abs(np.cos(pitch_guess)) > 1e-6 else 0.0
        R = np.array([r0, r1])  # [r0, r1]
        for it in range(max_iter):
            r0, r1 = R[0], R[1]
            # 计算常数 c (依赖于 r1)
            c = (self.g * (1 + r1**2) / (self.k * v0**2) +
                 r1 * np.sqrt(1 + r1**2) +
                 np.log(r1 + np.sqrt(1 + r1**2)))
            # 计算当前位移
            X = self._integral_X(r0, r1, c)
            Y = self._integral_Y(r0, r1, c)
            # 残差
            D = np.array([x0 - X, y0 - Y])
            if np.linalg.norm(D) < tol:
                pitch = np.arctan(r1)
                fly_time = self._integral_time(r0, r1, c)
                return fly_time, pitch
            # 数值计算雅可比矩阵
            eps = 1e-6
            # 对 r0 的偏导（c 不变）
            X_r0 = (self._integral_X(r0 + eps, r1, c) - X) / eps
            Y_r0 = (self._integral_Y(r0 + eps, r1, c) - Y) / eps
            # 对 r1 的偏导（需要重新计算 c1）
            r1_eps = r1 + eps
            c1 = (self.g * (1 + r1_eps**2) / (self.k * v0**2) +
                  r1_eps * np.sqrt(1 + r1_eps**2) +
                  np.log(r1_eps + np.sqrt(1 + r1_eps**2)))
            X1 = self._integral_X(r0, r1_eps, c1)
            Y1 = self._integral_Y(r0, r1_eps, c1)
            X_r1 = (X1 - X) / eps
            Y_r1 = (Y1 - Y) / eps
            J = np.array([[X_r0, X_r1],
                          [Y_r0, Y_r1]])
            # 求解增量
            cond = np.linalg.cond(J)
            if cond > 1e12:
                dR = np.linalg.pinv(J) @ D
            else:
                try:
                    dR = np.linalg.solve(J, D)
                except np.linalg.LinAlgError:
                    dR = np.linalg.pinv(J) @ D
            # 限制步长防止发散
            if np.linalg.norm(dR) > 1.0:
                dR = dR / np.linalg.norm(dR) * 1.0
            R += dR
            # 防止 r1 过大（避免数值问题）
            if abs(R[1]) > 10:
                R[1] = np.clip(R[1], -10, 10)
        print("TrajectorySolver: Failed to converge.")
        return None, None

    def _integral_X(self, r0, r1, c):
        dr = (r0 - r1) / self.N
        total = 0.0
        for i in range(self.N):
            r = r1 + (i + 0.5) * dr
            denom = self.k * (c - r * np.sqrt(1 + r**2) - np.log(r + np.sqrt(1 + r**2)))
            total += -1.0 / denom
        return total * dr

    def _integral_Y(self, r0, r1, c):
        dr = (r0 - r1) / self.N
        total = 0.0
        for i in range(self.N):
            r = r1 + (i + 0.5) * dr
            denom = self.k * (c - r * np.sqrt(1 + r**2) - np.log(r + np.sqrt(1 + r**2)))
            total += -r / denom
        return total * dr

    def _integral_time(self, r0, r1, c):
        dr = (r0 - r1) / self.N
        total = 0.0
        for i in range(self.N):
            r = r1 + (i + 0.5) * dr
            denom = np.sqrt(self.g * self.k * (c - r * np.sqrt(1 + r**2) - np.log(r + np.sqrt(1 + r**2))))
            total += -1.0 / denom
        return total * dr

