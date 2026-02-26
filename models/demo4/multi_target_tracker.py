import numpy as np
from models.demo4.demo_model_4 import DemoModel4

class TrackedTarget:
    def __init__(self, target_id, init_obs, t):
        self.id = target_id
        self.ekf = DemoModel4()
        self.ekf.init_model(init_obs.world_pos, init_obs.world_rpy[2], t)
        self.last_update = t
        self.update_count = 1
        self.robot_type = init_obs.robot_type
        self.matched_armors_count = 0

    def predict(self, dt):
        if dt > 0:
            self.ekf.predict(dt)

    def update(self, obs, t):
        # 外部已调用predict，此处只做更新
        self.ekf.update(obs.world_pos, obs.world_rpy[2], t)
        self.last_update = t
        self.update_count += 1


class MultiTargetTracker:
    def __init__(self, max_lost_time=0.5, match_threshold=0.4, camera_manager=None):
        self.targets = []
        self.next_id = 0
        self.max_lost_time = max_lost_time
        self.match_threshold = match_threshold
        self.camera_manager = camera_manager

    def push_observation(self, obs_list, t):
        # 1. 预测所有现有目标到当前时刻
        for target in self.targets:
            dt = t - target.last_update
            target.predict(dt)
            target.matched_armors_count = 0

        # 2. 数据关联（贪婪匹配）
        matches = {j: [] for j in range(len(self.targets))}
        for i, obs in enumerate(obs_list):
            best_target_idx = None
            min_dist = self.match_threshold
            for j, target in enumerate(self.targets):
                if target.robot_type != obs.robot_type:
                    continue
                # 使用几何距离（观测点到目标四个理论装甲板的最短距离）
                dist = target.ekf.get_geometric_distance(obs.world_pos)
                if dist < min_dist:
                    min_dist = dist
                    best_target_idx = j
            if best_target_idx is not None:
                matches[best_target_idx].append(i)

        # 3. 用匹配的观测更新对应目标
        used_obs_indices = set()
        for j, obs_indices in matches.items():
            target = self.targets[j]
            for i in obs_indices:
                obs = obs_list[i]
                target.update(obs, t)
                target.matched_armors_count += 1
                used_obs_indices.add(i)

        # 4. 未匹配的观测创建新目标
        for i in range(len(obs_list)):
            if i not in used_obs_indices:
                self._create_target(obs_list[i], t)

        # 5. 清理超时目标
        self._cleanup_targets(t)

    def _create_target(self, obs, t):
        new_target = TrackedTarget(self.next_id, obs, t)
        self.targets.append(new_target)
        self.next_id += 1

    def _cleanup_targets(self, t):
        active_targets = []
        for tgt in self.targets:
            if (t - tgt.last_update) < self.max_lost_time:
                active_targets.append(tgt)
        self.targets = active_targets

    def get_best_target(self):
        if not self.targets:
            return None
        # 按更新次数排序，选择最稳定的目标
        sorted_targets = sorted(
            self.targets,
            key=lambda t: t.update_count,
            reverse=True
        )
        return sorted_targets[0]