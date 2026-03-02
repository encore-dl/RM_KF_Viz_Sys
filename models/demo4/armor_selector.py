import numpy as np
from core.algorithms.math import limit_rad


class ArmorSelector:
    def __init__(self, spin_thresh=12.0, lock_bias=0.3):
        self.spin_thresh = spin_thresh
        self.lock_bias = lock_bias
        self.locked_armor_id = None

    def select_armor(self, target, self_rel_pos, t_future):
        if target.jumped:
            self.locked_armor_id = None

        armors_pos = target.ekf.get_all_armor_positions_at_time(t_future)
        x = target.ekf.ekf.x
        omega = x[7]

        if abs(omega) >= self.spin_thresh:
            # 高速旋转：选择指向自车方向的装甲板
            cx = x[0] + x[3] * t_future
            cy = x[1] + x[4] * t_future
            dx_self = self_rel_pos[0] - cx
            dy_self = self_rel_pos[1] - cy
            yaw_to_self = np.arctan2(dy_self, dx_self)

            best_id = 0
            min_diff = float('inf')
            for i, pos in enumerate(armors_pos):
                dx_armor = pos[0] - cx
                dy_armor = pos[1] - cy
                armor_yaw = np.arctan2(dy_armor, dx_armor)
                diff = abs(limit_rad(armor_yaw - yaw_to_self))
                if diff < min_diff:
                    min_diff = diff
                    best_id = i
            return best_id, armors_pos[best_id]
        else:
            # 低速旋转：根据可视性选择
            best_score = -np.inf
            best_id = None
            best_pos = None
            for i, pos in enumerate(armors_pos):
                psi = x[6] + x[7] * t_future
                armor_yaw = psi + i * np.pi / 2
                normal = np.array([np.cos(armor_yaw), np.sin(armor_yaw), 0])
                sight = self_rel_pos - pos
                dist = np.linalg.norm(sight)
                if dist < 1e-6:
                    continue
                sight_unit = sight / dist
                cos_alpha = np.dot(normal, sight_unit)
                if cos_alpha <= 0:
                    continue
                score = cos_alpha / (dist * dist)
                if self.locked_armor_id is not None and i == self.locked_armor_id:
                    score += self.lock_bias
                if score > best_score:
                    best_score = score
                    best_id = i
                    best_pos = pos

            if best_id is not None:
                self.locked_armor_id = best_id
                return best_id, best_pos
            else:
                self.locked_armor_id = None
                return 0, armors_pos[0]