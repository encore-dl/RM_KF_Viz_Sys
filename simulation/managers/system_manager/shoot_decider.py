import numpy as np
import math
from simulation.event_bus import event_bus
from core.algorithms.trajectory.trajectory_solver import TrajectorySolver

class ShootDecider:
    def __init__(self, camera_manager, v0=30.0, fire_threshold=0.05, cooldown=0.1):
        self.camera_manager = camera_manager
        self.v0 = v0
        self.fire_threshold = fire_threshold
        self.cooldown = cooldown
        self.last_fire_time = 0
        self.traj_solver = TrajectorySolver(k=0.001)   # 阻力系数可调

    def update(self, target_ekf, current_time):
        if target_ekf is None or not target_ekf.is_init:
            return False

        camera = self.camera_manager.selected_camera
        gun_pos = camera.world_pos

        # 获取目标中心当前位置（作为迭代初始）
        center_pos = target_ekf.ekf.x[:3]
        vec = center_pos - gun_pos
        dist = np.linalg.norm(vec)
        if dist < 1e-6:
            return False

        # 初始猜测：当前 pitch
        dx = center_pos[0] - gun_pos[0]
        dy = center_pos[1] - gun_pos[1]
        dz = center_pos[2] - gun_pos[2]
        pitch_guess = np.arctan2(dz, np.sqrt(dx*dx + dy*dy))

        # 迭代求解弹道（最多 5 次）
        fly_time = dist / self.v0
        best_pitch = pitch_guess
        best_armor = None
        for _ in range(5):
            # 预测目标在 fly_time 后的位置
            future_armors = target_ekf.get_all_armor_positions_at_time(fly_time)
            # 选择最佳装甲板（简化：取第一个可见的）
            best_score = -np.inf
            for k, pos in enumerate(future_armors):
                pred_psi = target_ekf.ekf.x[6] + target_ekf.ekf.x[7] * fly_time
                normal = np.array([np.cos(pred_psi + k*np.pi/2), np.sin(pred_psi + k*np.pi/2), 0])
                sight = gun_pos - pos
                dist_armor = np.linalg.norm(sight)
                if dist_armor < 1e-6:
                    continue
                cos_alpha = np.dot(normal, sight / dist_armor)
                if cos_alpha <= 0:
                    continue
                score = cos_alpha / (dist_armor * dist_armor)
                if score > best_score:
                    best_score = score
                    best_armor = pos
            if best_armor is None:
                best_armor = future_armors[0]   # 保底

            # 用弹道模型重新计算飞行时间和 pitch
            target_rel = best_armor - gun_pos
            fly_time_new, pitch_new = self.traj_solver.solve(self.v0, target_rel, best_pitch)
            if fly_time_new is None:
                break
            if abs(fly_time_new - fly_time) < 0.001:
                fly_time = fly_time_new
                best_pitch = pitch_new
                break
            fly_time = fly_time_new
            best_pitch = pitch_new

        if fly_time is None:
            return False

        # 最终期望角度
        dx = best_armor[0] - gun_pos[0]
        dy = best_armor[1] - gun_pos[1]
        dz = best_armor[2] - gun_pos[2]
        desired_yaw = np.arctan2(dy, dx)
        desired_pitch = best_pitch

        current_yaw = camera.world_rpy[2]
        current_pitch = camera.world_rpy[1]

        yaw_diff = (desired_yaw - current_yaw + np.pi) % (2*np.pi) - np.pi
        pitch_diff = desired_pitch - current_pitch

        if abs(yaw_diff) > self.fire_threshold or abs(pitch_diff) > self.fire_threshold:
            return False
        if current_time - self.last_fire_time < self.cooldown:
            return False

        self._fire()
        self.last_fire_time = current_time
        return True

    def _fire(self):
        """发射子弹，方向取当前云台朝向"""
        camera = self.camera_manager.selected_camera
        yaw = camera.world_rpy[2]
        pitch = camera.world_rpy[1]
        muzzle_pos = camera.world_pos.copy()
        vel = self.v0 * np.array([
            math.cos(yaw) * math.cos(pitch),
            math.sin(yaw) * math.cos(pitch),
            math.sin(pitch)
        ])
        event_bus.publish('fire', {'pos': muzzle_pos, 'vel': vel})