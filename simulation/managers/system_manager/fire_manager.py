import numpy as np
from simulation.event_bus import event_bus

class FireManager:
    def __init__(self, camera, fire_threshold=0.05, fire_cooldown=0.1, muzzle_vel=30.0):
        self.camera = camera
        self.fire_threshold = fire_threshold
        self.fire_cooldown = fire_cooldown
        self.muzzle_vel = muzzle_vel
        self.last_fire_time = 0

    def update(self, target_ekf, current_yaw, current_time):
        """
        根据目标状态和当前角度决定是否开火
        返回 True 如果开火，否则 False
        """
        # 这里使用目标中心计算理想瞄准角度（简化）
        # 也可以使用参考轨迹的第一个点，此处简单处理
        if target_ekf is None or not target_ekf.is_init:
            return False
        target_pos = target_ekf.ekf.x[:3]  # 当前中心位置
        dx = target_pos[0] - self.camera.world_pos[0]
        dy = target_pos[1] - self.camera.world_pos[1]
        desired_yaw = np.arctan2(dy, dx)
        # 计算最短角度差
        diff = desired_yaw - current_yaw
        diff = (diff + np.pi) % (2 * np.pi) - np.pi

        if abs(diff) > self.fire_threshold:
            return False
        if current_time - self.last_fire_time < self.fire_cooldown:
            return False
        self._fire()
        self.last_fire_time = current_time
        return True

    def _fire(self):
        yaw = self.camera.world_rpy[2]
        pitch = self.camera.world_rpy[1]
        muzzle_pos = self.camera.world_pos.copy()
        vel = self.muzzle_vel * np.array([
            np.cos(yaw) * np.cos(pitch),
            np.sin(yaw) * np.cos(pitch),
            np.sin(pitch)
        ])
        event_bus.publish('fire', {'pos': muzzle_pos, 'vel': vel})