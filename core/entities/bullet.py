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


