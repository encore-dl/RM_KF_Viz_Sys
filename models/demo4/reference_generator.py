import numpy as np

from core.algorithms.math import limit_rad
from simulation.event_bus import event_bus
from simulation.dataflow import DrawText

class ReferenceGenerator:
    def __init__(self, camera, dt=0.01, N=20, yaw_offset=0.0, pitch_offset=0.0):
        self.camera = camera
        self.dt = dt
        self.N = N
        self.yaw_offset = yaw_offset
        self.pitch_offset = pitch_offset

    def generate(self, target_ekf, t0):
        """
        生成未来 N 步的期望云台角度 和 角速度
        """
        theta_ref = np.zeros(self.N)
        omega_ref = np.zeros(self.N)  # 新增角速度参考
        phi_ref = np.zeros(self.N)

        cam_pos = self.camera.world_pos

        # 缓存一下状态，避免循环里频繁访问
        x_curr = target_ekf.ekf.x.copy()

        for i in range(self.N):
            dt = i * self.dt

            # 1. 外推状态 (CV 模型)
            # 这里的 x, y 是装甲板中心坐标
            pred_xc = x_curr[0] + x_curr[3] * dt
            pred_yc = x_curr[1] + x_curr[4] * dt
            pred_zc = x_curr[2] + x_curr[5] * dt

            # 速度 (CV模型假设速度不变)
            pred_vx = x_curr[3]
            pred_vy = x_curr[4]
            pred_vz = x_curr[5]

            # 旋转预测
            pred_psi = x_curr[6] + x_curr[7] * dt

            # 2. 选装甲板 (简化逻辑：直接找最近的)
            # 为了计算平滑，这里我们尽量使用中心点或者选定的装甲板
            # 这里为了简单，我们计算 "面向相机的最佳击打点" 的参考轨迹
            # 这样更稳，防止装甲板切换导致的参考轨迹跳变

            # 计算目标相对于相机的向量
            dx = pred_xc - cam_pos[0]
            dy = pred_yc - cam_pos[1]
            dz = pred_zc - cam_pos[2]

            # 相对速度
            dvx = pred_vx  # 假设相机不动，如果相机动需减去相机速度
            dvy = pred_vy

            # 3. 计算期望 Yaw (Theta)
            theta_des = np.arctan2(dy, dx)

            # 4. 【核心】解析计算期望角速度 (Omega)
            # d(atan2(y, x)) = (x*dy - y*dx) / (x^2 + y^2)
            # 这个公式在 x负半轴是连续的！
            r2 = dx * dx + dy * dy
            if r2 > 1e-6:
                omega_des = (dx * dvy - dy * dvx) / r2
            else:
                omega_des = 0.0

            # Pitch 计算
            dist_xy = np.sqrt(r2)
            phi_des = np.arctan2(dz, dist_xy)

            theta_ref[i] = theta_des + self.yaw_offset
            omega_ref[i] = omega_des  # 存储角速度
            phi_ref[i] = phi_des + self.pitch_offset

        # 位置需要解缠，速度不需要（速度是物理量，没有相位问题）
        theta_ref = self._unwrap_angle(theta_ref)

        return theta_ref, omega_ref, phi_ref  # 返回三个

    def _select_best_armor(self, x, cam_pos):
        """选择距离最近的装甲板"""
        best_k = None
        best_dist = float('inf')
        for k in range(4):
            is_even = (k % 2 == 0)
            r = x[8] if is_even else x[9]
            dz = x[10] if not is_even else 0.0
            armor_yaw = x[6] + k * np.pi / 2.0
            armor_pos = np.array([
                x[0] + r * np.cos(armor_yaw),
                x[1] + r * np.sin(armor_yaw),
                x[2] + dz
            ])
            d = np.linalg.norm(armor_pos - cam_pos)
            if d < best_dist:
                best_dist = d
                best_k = k
        return best_k, best_dist

    @staticmethod
    def _unwrap_angle(angle_seq):
        unwrapped = np.zeros_like(angle_seq)
        unwrapped[0] = angle_seq[0]
        for i in range(1, len(angle_seq)):
            diff = angle_seq[i] - unwrapped[i - 1]
            if diff > np.pi:
                diff -= 2 * np.pi
            elif diff < -np.pi:
                diff += 2 * np.pi
            unwrapped[i] = unwrapped[i - 1] + diff
        return unwrapped