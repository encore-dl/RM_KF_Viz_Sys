import numpy as np

from core.algorithms.math import limit_rad


class ReferenceGenerator:
    def __init__(self, camera_manager, dt=0.01, N=20, yaw_offset=0.0, pitch_offset=0.0, spin_thresh=2.0):
        self.camera_manager = camera_manager
        self.dt = dt
        self.N = N
        self.yaw_offset = yaw_offset
        self.pitch_offset = pitch_offset
        self.spin_thresh = spin_thresh

    def generate(self, target_ekf, t0):
        theta_ref = np.zeros(self.N)
        omega_ref = np.zeros(self.N)
        phi_ref = np.zeros(self.N)

        cam_pos = self.camera_manager.selected_camera.world_pos
        x_curr = target_ekf.ekf.x.copy()
        current_omega = x_curr[7]  # 当前角速度

        for i in range(self.N):
            dt = i * self.dt
            # 外推状态
            pred_xc = x_curr[0] + x_curr[3] * dt
            pred_yc = x_curr[1] + x_curr[4] * dt
            pred_zc = x_curr[2] + x_curr[5] * dt
            pred_psi = x_curr[6] + x_curr[7] * dt
            ra, rb, dz = x_curr[8], x_curr[9], x_curr[10]

            if abs(current_omega) < self.spin_thresh:
                # 慢速旋转：选择最佳装甲板
                armors_pos = []
                for k in range(4):
                    is_even = (k % 2 == 0)
                    r = ra if is_even else rb
                    h = 0.0 if is_even else dz
                    armor_yaw = pred_psi + k * np.pi / 2.0
                    ax = pred_xc + r * np.cos(armor_yaw)
                    ay = pred_yc + r * np.sin(armor_yaw)
                    az = pred_zc + h
                    armors_pos.append(np.array([ax, ay, az]))

                best_armor_pos = None
                best_score = -np.inf
                for k, pos in enumerate(armors_pos):
                    armor_yaw = pred_psi + k * np.pi / 2.0
                    normal = np.array([np.cos(armor_yaw), np.sin(armor_yaw), 0])
                    sight = cam_pos - pos
                    dist = np.linalg.norm(sight)
                    if dist < 1e-6:
                        continue
                    sight_unit = sight / dist
                    cos_alpha = np.dot(normal, sight_unit)
                    if cos_alpha <= 0:
                        continue
                    score = cos_alpha / (dist * dist)
                    if score > best_score:
                        best_score = score
                        best_armor_pos = pos

                if best_armor_pos is None:
                    # 没有可见装甲板时，使用面向枪口的点（快速模式）
                    yaw_to_self = np.arctan2(-pred_yc, -pred_xc)
                    armor_yaw = yaw_to_self
                    best_armor_pos = np.array([
                        pred_xc + ra * np.cos(armor_yaw),
                        pred_yc + ra * np.sin(armor_yaw),
                        pred_zc
                    ])  # 简化，使用平均半径
            else:
                # 快速旋转：瞄准面向枪口的点
                yaw_to_self = np.arctan2(-pred_yc, -pred_xc)
                # 使用平均半径（或 ra）计算装甲板位置
                r_mean = (ra + rb) / 2
                armor_yaw = yaw_to_self
                best_armor_pos = np.array([
                    pred_xc + r_mean * np.cos(armor_yaw),
                    pred_yc + r_mean * np.sin(armor_yaw),
                    pred_zc
                ])

            # 计算期望角度
            dx = best_armor_pos[0] - cam_pos[0]
            dy = best_armor_pos[1] - cam_pos[1]
            dz = best_armor_pos[2] - cam_pos[2]
            theta_des = np.arctan2(dy, dx)
            dist_xy = np.sqrt(dx*dx + dy*dy)
            phi_des = np.arctan2(dz, dist_xy)

            # 角速度（仍使用目标中心速度近似）
            pred_vx = x_curr[3]
            pred_vy = x_curr[4]
            r2 = dx*dx + dy*dy
            if r2 > 1e-6:
                omega_des = (dx * pred_vy - dy * pred_vx) / r2
            else:
                omega_des = 0.0

            theta_ref[i] = theta_des + self.yaw_offset
            omega_ref[i] = omega_des
            phi_ref[i] = phi_des + self.pitch_offset

        # 角度解缠
        theta_ref = self._unwrap_angle(theta_ref)
        return theta_ref, omega_ref, phi_ref

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