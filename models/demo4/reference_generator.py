import numpy as np


class ReferenceGenerator:
    def __init__(self, robot_manager, dt=0.01, N=50, yaw_offset=0.0, pitch_offset=0.0,
                 spin_thresh=12.0, lock_bias=0.2):
        self.robot_manager = robot_manager
        self.dt = dt
        self.N = N
        self.yaw_offset = yaw_offset
        self.pitch_offset = pitch_offset
        self.spin_thresh = spin_thresh
        self.lock_bias = lock_bias
        self.locked_armor_id = None

    def generate(self, target_ekf, t0):
        theta_ref = np.zeros(self.N)
        omega_ref = np.zeros(self.N)
        phi_ref = np.zeros(self.N)
        phi_omega_ref = np.zeros(self.N)

        if self.robot_manager.selected_robot is None:
            return theta_ref, omega_ref, phi_ref, phi_omega_ref

        camera = self.robot_manager.selected_robot.get_camera()
        cam_pos = camera.world_pos
        x_curr = target_ekf.ekf.x.copy()
        current_omega = x_curr[7]

        for i in range(self.N):
            dt = i * self.dt
            pred_xc = x_curr[0] + x_curr[3] * dt
            pred_yc = x_curr[1] + x_curr[4] * dt
            pred_zc = x_curr[2] + x_curr[5] * dt
            pred_psi = x_curr[6] + x_curr[7] * dt
            ra, rb, dz = x_curr[8], x_curr[9], x_curr[10]

            if abs(current_omega) < self.spin_thresh:
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
                best_id = None
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
                    if self.locked_armor_id is not None and k == self.locked_armor_id:
                        score += self.lock_bias
                    if score > best_score:
                        best_score = score
                        best_armor_pos = pos
                        best_id = k

                if best_id is not None:
                    self.locked_armor_id = best_id
                else:
                    self.locked_armor_id = None

                if best_armor_pos is None:
                    yaw_to_self = np.arctan2(-pred_yc, -pred_xc)
                    armor_yaw = yaw_to_self
                    best_armor_pos = np.array([
                        pred_xc + ra * np.cos(armor_yaw),
                        pred_yc + ra * np.sin(armor_yaw),
                        pred_zc
                    ])
                    self.locked_armor_id = None
            else:
                self.locked_armor_id = None
                yaw_to_self = np.arctan2(-pred_yc, -pred_xc)
                r_mean = (ra + rb) / 2
                armor_yaw = yaw_to_self
                best_armor_pos = np.array([
                    pred_xc + r_mean * np.cos(armor_yaw),
                    pred_yc + r_mean * np.sin(armor_yaw),
                    pred_zc
                ])

            dx = best_armor_pos[0] - cam_pos[0]
            dy = best_armor_pos[1] - cam_pos[1]
            dz = best_armor_pos[2] - cam_pos[2]
            theta_des = np.arctan2(dy, dx)
            dist_xy = np.sqrt(dx*dx + dy*dy)
            phi_des = -np.arctan2(dz, dist_xy)

            theta_ref[i] = theta_des + self.yaw_offset
            phi_ref[i] = phi_des + self.pitch_offset

            pred_vx = x_curr[3]
            pred_vy = x_curr[4]
            r2 = dx*dx + dy*dy
            if r2 > 1e-6:
                omega_des = (dx * pred_vy - dy * pred_vx) / r2
            else:
                omega_des = 0.0
            omega_ref[i] = omega_des

        for i in range(1, self.N - 1):
            phi_omega_ref[i] = (phi_ref[i + 1] - phi_ref[i - 1]) / (2 * self.dt)
        if self.N > 1:
            phi_omega_ref[0] = (phi_ref[1] - phi_ref[0]) / self.dt
            phi_omega_ref[-1] = (phi_ref[-1] - phi_ref[-2]) / self.dt

        theta_ref = self._unwrap_angle(theta_ref)
        return theta_ref, omega_ref, phi_ref, phi_omega_ref

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