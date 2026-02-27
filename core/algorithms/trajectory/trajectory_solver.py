import numpy as np

class TrajectorySolver:
    def __init__(self, g=9.8, k=0.001, N=50):
        self.g = g
        self.k = k
        self.N = N

    def solve(self, v0, target_pos, pitch_guess, max_iter=20, tol=1e-5):
        x0 = np.sqrt(target_pos[0]**2 + target_pos[1]**2)
        y0 = target_pos[2]
        r1 = np.tan(pitch_guess)
        t_est = x0 / (v0 * np.cos(pitch_guess))
        r0 = (v0 * np.sin(pitch_guess) - self.g * t_est) / (v0 * np.cos(pitch_guess))
        R = np.array([r0, r1])
        for it in range(max_iter):
            r0, r1 = R[0], R[1]
            c = (self.g * (1 + r1**2) / (self.k * v0**2) +
                 r1 * np.sqrt(1 + r1**2) +
                 np.log(r1 + np.sqrt(1 + r1**2)))
            X = self._integral_X(r0, r1, c)
            Y = self._integral_Y(r0, r1, c)
            D = np.array([x0 - X, y0 - Y])
            if np.linalg.norm(D) < tol:
                pitch = np.arctan(r1)
                fly_time = self._integral_time(r0, r1, c) / np.sqrt(self.g * self.k)
                return fly_time, pitch
            eps = 1e-6
            X_r0 = (self._integral_X(r0 + eps, r1, c) - X) / eps
            Y_r0 = (self._integral_Y(r0 + eps, r1, c) - Y) / eps
            X_r1 = (self._integral_X(r0, r1 + eps, c) - X) / eps
            Y_r1 = (self._integral_Y(r0, r1 + eps, c) - Y) / eps
            J = np.array([[X_r0, X_r1], [Y_r0, Y_r1]])
            cond = np.linalg.cond(J)
            if cond > 1e12:
                dR = np.linalg.pinv(J) @ D
            else:
                try:
                    dR = np.linalg.solve(J, D)
                except np.linalg.LinAlgError:
                    dR = np.linalg.pinv(J) @ D
            if np.linalg.norm(dR) > 1.0:
                dR = dR / np.linalg.norm(dR) * 1.0
            R += dR
            if abs(r1) > 10:
                R[1] = np.clip(r1, -10, 10)
        print("Failed to converge.")
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