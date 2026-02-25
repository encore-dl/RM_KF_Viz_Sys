import numpy as np
from simulation.event_bus import event_bus


class ShootDecider:
    def __init__(self, camera_manager, v0=10., fire_threshold=0.05, cooldown=0.1):
        """
        射击决策器
        :param camera_manager: 相机（用于获取当前位置和姿态）
        :param v0: 子弹初速 (m/s)
        :param fire_threshold: 角度误差阈值 (rad)
        :param cooldown: 开火冷却时间 (s)
        """
        self.camera_manager = camera_manager
        self.v0 = v0
        self.fire_threshold = fire_threshold
        self.cooldown = cooldown
        self.last_fire_time = 0

    def update(self, target_ekf, current_time):
        """
        根据目标EKF状态决定是否开火
        :param target_ekf: DemoModel4 实例
        :param current_time: 当前时间戳
        :return: bool 是否开火
        """
        if target_ekf is None or not target_ekf.is_init:
            return False

        # 1. 估算子弹飞行时间（简化：用当前距离除以初速）
        #    获取目标中心当前位置
        center_pos = target_ekf.ekf.x[:3]
        gun_pos = self.camera_manager.selected_camera.world_pos
        vec = center_pos - gun_pos
        dist = np.linalg.norm(vec)
        if dist < 1e-6:
            return False
        t_fly = dist / self.v0

        # 2. 预测 t_fly 后的机器人状态和所有装甲板位置
        future_armors = target_ekf.get_all_armor_positions_at_time(t_fly)

        # 3. 选择最佳装甲板
        best_armor_pos = None
        best_score = -np.inf
        gun_pos = self.camera_manager.selected_camera.world_pos

        for k, armor_pos in enumerate(future_armors):
            # 计算装甲板法向量（指向机器人中心）
            # 注意：预测时机器人角度已变化，需要重新获取角度
            # 简便方法：从 target_ekf 获取预测后的角度（已包含在 get_all... 内部）
            # 但我们需要法向量，需要知道预测后的 yaw
            # 可以在 get_all_armor_positions_at_time 中同时返回法向量，或者这里重新计算
            # 这里重新计算角度
            x = target_ekf.ekf.x
            pred_psi = x[6] + x[7] * t_fly
            armor_yaw = pred_psi + k * np.pi / 2.0
            normal = np.array([np.cos(armor_yaw), np.sin(armor_yaw), 0])  # 指向中心

            # 视线向量（从装甲板指向相机）或者从相机指向装甲板？法向量指向中心，所以视线向量应为相机位置 - 装甲板位置
            sight = gun_pos - armor_pos
            sight_norm = np.linalg.norm(sight)
            if sight_norm < 1e-6:
                continue
            sight_unit = sight / sight_norm

            # 可见性判断：法向量与视线的点积 > 0 表示装甲板朝向相机
            cos_alpha = np.dot(normal, sight_unit)
            if cos_alpha <= 0:
                continue

            # 评分：距离越近越好，同时考虑朝向（可选）
            score = cos_alpha / (sight_norm * sight_norm)  # 避免除零
            if score > best_score:
                best_score = score
                best_armor_pos = armor_pos

        if best_armor_pos is None:
            return False

        # 4. 计算期望瞄准角度
        dx = best_armor_pos[0] - gun_pos[0]
        dy = best_armor_pos[1] - gun_pos[1]
        dz = best_armor_pos[2] - gun_pos[2]
        desired_yaw = np.arctan2(dy, dx)
        dist_xy = np.sqrt(dx*dx + dy*dy)
        desired_pitch = np.arctan2(dz, dist_xy)

        # 5. 计算当前云台角度
        current_yaw = self.camera_manager.selected_camera.world_rpy[2]
        current_pitch = self.camera_manager.selected_camera.world_rpy[1]

        # 角度差归一化
        yaw_diff = desired_yaw - current_yaw
        yaw_diff = (yaw_diff + np.pi) % (2*np.pi) - np.pi
        pitch_diff = desired_pitch - current_pitch
        pitch_diff = (pitch_diff + np.pi) % (2*np.pi) - np.pi

        # 6. 判断是否开火
        if abs(yaw_diff) > self.fire_threshold or abs(pitch_diff) > self.fire_threshold:
            return False
        if current_time - self.last_fire_time < self.cooldown:
            return False

        # 7. 开火
        self._fire()
        self.last_fire_time = current_time
        return True

    def _fire(self):
        """发射子弹，方向取当前云台朝向"""
        yaw = self.camera_manager.selected_camera.world_rpy[2]
        pitch = self.camera_manager.selected_camera.world_rpy[1]
        muzzle_pos = self.camera_manager.selected_camera.world_pos.copy()
        vel = self.v0 * np.array([
            np.cos(yaw) * np.cos(pitch),
            np.sin(yaw) * np.cos(pitch),
            np.sin(pitch)
        ])
        event_bus.publish('fire', {'pos': muzzle_pos, 'vel': vel})