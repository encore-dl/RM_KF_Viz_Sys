import numpy as np

class TrajectorySolver:
    def __init__(self, g=9.8, k=0.001, N=50):
        self.g = g
        self.k = k
        self.N = N          # 积分步数（提高精度）

    def solve(self, v0, target_pos, pitch_guess, max_iter=20, tol=1e-5):
        """
        迭代求解满足目标位置的发射角和飞行时间
        :param v0: 子弹初速
        :param target_pos: 目标相对发射点的位置 (x, y, z) 世界坐标系
        :param pitch_guess: 初始俯仰角猜测 (rad)
        :param max_iter: 最大迭代次数
        :param tol: 收敛容差
        :return: (fly_time, pitch) 或 (None, None) 若失败
        """
        x0 = np.sqrt(target_pos[0]**2 + target_pos[1]**2)  # 水平距离
        y0 = target_pos[2]                                  # 高度差

        # 初始猜测 r1 = tan(pitch_guess)
        r1 = np.tan(pitch_guess)
        # 初始猜测 r0 由飞行时间估算：忽略阻力，用抛物线
        t_est = x0 / (v0 * np.cos(pitch_guess))
        r0 = (v0 * np.sin(pitch_guess) - self.g * t_est) / (v0 * np.cos(pitch_guess))

        R = np.array([r0, r1])  # [r0, r1]

        for it in range(max_iter):
            r0, r1 = R[0], R[1]
            # 计算积分常数 c
            c = (self.g * (1 + r1**2) / (self.k * v0**2) +
                 r1 * np.sqrt(1 + r1**2) +
                 np.log(r1 + np.sqrt(1 + r1**2)))

            # 计算残差 D = [x0 - X, y0 - Y]
            X = self._integral_X(r0, r1, c)
            Y = self._integral_Y(r0, r1, c)
            D = np.array([x0 - X, y0 - Y])

            if np.linalg.norm(D) < tol:
                # 收敛
                pitch = np.arctan(r1)
                fly_time = self._integral_time(r0, r1, c) / np.sqrt(self.g * self.k)
                return fly_time, pitch

            # 计算雅可比矩阵（数值微分）
            eps = 1e-6
            X_r0 = (self._integral_X(r0 + eps, r1, c) - X) / eps
            Y_r0 = (self._integral_Y(r0 + eps, r1, c) - Y) / eps
            X_r1 = (self._integral_X(r0, r1 + eps, c) - X) / eps
            Y_r1 = (self._integral_Y(r0, r1 + eps, c) - Y) / eps
            J = np.array([[X_r0, X_r1], [Y_r0, Y_r1]])

            # 解线性方程组 J * dR = D
            try:
                dR = np.linalg.solve(J, D)
            except np.linalg.LinAlgError:
                print("Jacobian singular, breaking.")
                break
            R += dR

        print("Failed to converge.")
        return None, None

    def _integral_X(self, r0, r1, c):
        a, b = r1, r0
        if a > b:
            a, b = b, a
        r_vals = np.linspace(a + 0.5 * (b - a) / self.N, b - 0.5 * (b - a) / self.N, self.N)
        denom = self.k * (c - r_vals * np.sqrt(1 + r_vals**2) - np.log(r_vals + np.sqrt(1 + r_vals**2)))
        integrand = -1.0 / denom
        return np.sum(integrand) * (b - a) / self.N

    def _integral_Y(self, r0, r1, c):
        a, b = r1, r0
        if a > b:
            a, b = b, a
        r_vals = np.linspace(a + 0.5 * (b - a) / self.N, b - 0.5 * (b - a) / self.N, self.N)
        denom = self.k * (c - r_vals * np.sqrt(1 + r_vals**2) - np.log(r_vals + np.sqrt(1 + r_vals**2)))
        integrand = -r_vals / denom
        return np.sum(integrand) * (b - a) / self.N

    def _integral_time(self, r0, r1, c):
        a, b = r1, r0
        if a > b:
            a, b = b, a
        r_vals = np.linspace(a + 0.5 * (b - a) / self.N, b - 0.5 * (b - a) / self.N, self.N)
        denom = np.sqrt(self.g * self.k * (c - r_vals * np.sqrt(1 + r_vals**2) - np.log(r_vals + np.sqrt(1 + r_vals**2))))
        integrand = -1.0 / denom
        return np.sum(integrand) * (b - a) / self.N