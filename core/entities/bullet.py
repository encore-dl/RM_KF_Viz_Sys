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

    def check_collision(self, armor):
        """
        检查子弹是否与装甲板碰撞
        简化：判断子弹到装甲板平面的距离 < 阈值，且投影点在四边形内
        """
        # 装甲板四个角点
        corners = armor.light_corners
        # 计算装甲板平面法向量（近似为装甲板的X轴方向）
        # 取 corners[0] -> corners[1] 为一边，corners[0] -> corners[3] 为另一边
        v1 = corners[1] - corners[0]
        v2 = corners[3] - corners[0]
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            return False
        normal = normal / norm
        # 计算子弹到平面的距离
        d = np.dot(self.pos - corners[0], normal)
        if abs(d) > 0.01:  # 10mm 阈值
            return False
        # 投影点到平面，判断是否在四边形内
        # 将点投影到平面
        proj = self.pos - d * normal
        # 转换为局部坐标（使用 v1, v2 作为基）
        # 解方程 proj = corners[0] + u * v1 + v * v2
        # 建立矩阵 M = [v1, v2] (3x2)
        M = np.column_stack((v1, v2))
        # 使用最小二乘解
        try:
            uv, _, _, _ = np.linalg.lstsq(M, proj - corners[0], rcond=None)
        except:
            return False
        u, v = uv[0], uv[1]
        # 检查是否在 [0,1] 范围内
        if 0 <= u <= 1 and 0 <= v <= 1:
            return True
        return False