import numpy as np
import math
import time

class Bullet:
    def __init__(self, pos, vel, mass=0.0032, caliber=6.8e-3, Cd=0.3, rho_air=1.2):
        """
        子弹物理模型
        pos: 初始位置 (世界坐标系)
        vel: 初始速度向量 (m/s)
        mass: 弹丸质量 (kg)
        caliber: 弹径 (m)
        Cd: 阻力系数
        rho_air: 空气密度 (kg/m^3)
        """
        self.pos = pos.copy()
        self.vel = vel.copy()
        self.mass = mass
        self.caliber = caliber
        self.Cd = Cd
        self.rho_air = rho_air
        self.A = math.pi * (caliber / 2) ** 2  # 截面积
        self.active = True
        self.birth_time = time.time()
        self.hit = False
        self.hit_pos = None
        self.hit_target = None
        self.hit_time = None

    def update(self, dt):
        """欧拉积分更新位置"""
        v = np.linalg.norm(self.vel)
        if v > 1e-6:
            F_drag = 0.5 * self.rho_air * self.A * self.Cd * v ** 2
            acc_drag = -F_drag / self.mass * (self.vel / v)
        else:
            acc_drag = 0
        acc_gravity = np.array([0, 0, -9.8])
        self.vel += (acc_drag + acc_gravity) * dt
        self.pos += self.vel * dt

    def check_segment_collision(self, prev_pos, curr_pos, armor):
        corners = armor.light_corners
        v1 = corners[1] - corners[0]
        v2 = corners[3] - corners[0]
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            return False, None
        normal = normal / norm

        d_prev = np.dot(prev_pos - corners[0], normal)
        d_curr = np.dot(curr_pos - corners[0], normal)
        if d_prev * d_curr > 0:
            return False, None

        t = -d_prev / (d_curr - d_prev)
        if t < 0 or t > 1:
            return False, None

        hit_point = prev_pos + t * (curr_pos - prev_pos)

        M = np.column_stack((v1, v2))
        try:
            uv, _, _, _ = np.linalg.lstsq(M, hit_point - corners[0], rcond=None)
        except:
            return False, None
        u, v = uv[0], uv[1]
        if not (0 <= u <= 1 and 0 <= v <= 1):
            return False, None

        # 检查子弹方向是否与法向量相反（正面击中）
        dir_vec = curr_pos - prev_pos
        dir_norm = np.linalg.norm(dir_vec)
        if dir_norm < 1e-6:
            return False, None
        dir_unit = dir_vec / dir_norm
        if np.dot(dir_unit, normal) >= 0:
            return False, None

        return True, hit_point

