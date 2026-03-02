import numpy as np
import time
from core.entities.bullet import Bullet
from simulation.event_bus import event_bus


class BulletManager:
    def __init__(self, robot_manager):
        self.robot_manager = robot_manager
        self.bullets = []
        self.pending_fires = []
        self.fire_delay = 0.1
        self.v0 = 10.
        event_bus.subscribe('fire', self._on_fire)

    def _on_fire(self, data):
        if not data.get('is_fire'):
            return

        robot = self.robot_manager.viewing_robot
        if robot is None:
            return
        # # 生成随机延迟（例如正态分布，均值为 0.1，标准差 0.05）
        # self.fire_delay = np.random.normal(0.1, 0.05)
        # # 确保延迟非负（截断到最小值 0）
        # self.fire_delay = max(self.fire_delay, 0.0)
        # 安排发射
        fire_time = time.time() + self.fire_delay
        self.pending_fires.append((fire_time, robot))

    def fire_bullet(self, muzzle_pos, velocity):
        """发射一颗子弹"""
        bullet = Bullet(muzzle_pos, velocity)
        self.bullets.append(bullet)
        return bullet

    def update(self, dt):
        """更新所有子弹，进行碰撞检测"""
        curr_t = time.time()
        # 检查待发射
        to_remove = []
        for idx, (fire_time, robot) in enumerate(self.pending_fires):
            if curr_t >= fire_time:
                muzzle = robot.get_muzzle()
                muzzle_pos = muzzle.world_pos.copy()
                muzzle_vel = muzzle.world_vel.copy()
                yaw = robot.gimbal.world_rpy[2]
                pitch = - robot.gimbal.world_rpy[1]
                direction = np.array([
                    np.cos(yaw) * np.cos(pitch),
                    np.sin(yaw) * np.cos(pitch),
                    np.sin(pitch)
                ])
                vel = muzzle_vel + self.v0 * direction
                self.fire_bullet(muzzle_pos, vel)
                to_remove.append(idx)
        for idx in reversed(to_remove):
            del self.pending_fires[idx]

        for bullet in self.bullets[:]:
            if bullet.hit:
                if curr_t - bullet.hit_time > 0.5:
                    self.bullets.remove(bullet)
                continue
            prev_pos = bullet.pos.copy()
            bullet.update(dt)
            curr_pos = bullet.pos
            if np.linalg.norm(curr_pos) > 50 or (time.time() - bullet.birth_time) > 5:
                bullet.active = False
                self.bullets.remove(bullet)
                continue
            hit_occurred = False
            for robot in self.robot_manager.robots:
                for armor in robot.get_armors():
                    hit, hit_pos = self.check_segment_collision(prev_pos, curr_pos, armor)
                    if hit:
                        bullet.hit = True
                        bullet.hit_pos = hit_pos
                        bullet.hit_target = (robot.robot_type, armor.armor_id)
                        bullet.hit_time = curr_t
                        hit_occurred = True
                        break
                if hit_occurred:
                    break

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

