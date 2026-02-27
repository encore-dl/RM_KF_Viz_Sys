import numpy as np
import time
from core.entities.bullet import Bullet


class BulletManager:
    def __init__(self, robot_manager):
        self.robot_manager = robot_manager
        self.bullets = []
        self.pending_fires = []

    def schedule_fire(self, muzzle_pos, velocity, delay):
        """计划在 delay 秒后发射子弹"""
        fire_time = time.time() + delay
        self.pending_fires.append((fire_time, muzzle_pos, velocity))

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
        for idx, (fire_time, pos, vel) in enumerate(self.pending_fires):
            if curr_t >= fire_time:
                self.fire_bullet(pos, vel)
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
                    hit, hit_pos = bullet.check_segment_collision(prev_pos, curr_pos, armor)
                    if hit:
                        bullet.hit = True
                        bullet.hit_pos = hit_pos
                        bullet.hit_target = (robot.robot_type, armor.armor_id)
                        bullet.hit_time = curr_t
                        hit_occurred = True
                        break
                if hit_occurred:
                    break


