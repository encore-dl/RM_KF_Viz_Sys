import numpy as np

from models.demo5.demo_model_5 import DemoModel5


class TrackedTarget:
    def __init__(self, init_obs, t):
        self.ekf = DemoModel5()
        self.ekf.init_model(init_obs)
        self.last_update = t
        self.update_count = 1
        self.robot_type = init_obs.robot_type
        self.last_match_id = 0
        self.jumped = False

        self.state = 'detecting'
        self.detect_count = 0
        self.lost_count = 0
        self.min_detect_count = 3
        self.max_lost_count = 10

        self.mahal_thresh_ = 9.49  # 马氏距离平方阈值（4维卡方95%分位）
        self.confirm_thresh_ = 3  # 新ID需要连续匹配成功多少帧才允许切换
        self.max_lost_ = 2  # 连续丢失多少帧后重置计数

    def predict(self, dt):
        if dt > 0:
            self.ekf.predict(dt)

    def update(self, obs, t):
        if self.ekf.diverged():
            print('diverged')
            self._reset()
            return

        self.ekf.update(obs)
        if self.ekf.match_id != self.last_match_id:
            self.jumped = True
        else:
            self.jumped = False
        self.last_match_id = self.ekf.match_id
        self.last_update = t
        self.update_count += 1

        if self.state == 'lost':
            self.state = 'detecting'
            self.detect_count = 1
            self.lost_count = 0
        elif self.state == 'detecting':
            self.detect_count += 1
            if self.detect_count >= self.min_detect_count:
                self.state = 'tracking'
                self.lost_count = 0
        elif self.state == 'tracking':
            self.lost_count = 0
        elif self.state == 'temp_lost':
            self.state = 'tracking'
            self.lost_count = 0

    def on_missing(self):
        if self.state == 'tracking':
            self.lost_count += 1
            if self.lost_count > self.max_lost_count:
                self.state = 'lost'
            else:
                self.state = 'temp_lost'
        elif self.state == 'detecting':
            self.state = 'lost'
        elif self.state == 'temp_lost':
            self.lost_count += 1
            if self.lost_count > self.max_lost_count:
                self.state = 'lost'

    def is_active(self):
        return self.state in ('detecting', 'tracking', 'temp_lost')

    def _reset(self):
        self.state = 'lost'
        self.detect_count = 0
        self.lost_count = 0
        self.last_match_id = 0
        self.ekf.is_init = False   # 强制重新初始化

    def get_pred_pos(self, dt, self_rel_pos, armor_selector):
        if not self.ekf.is_init:
            return np.zeros(3), np.zeros(3)
        x = self.ekf.ekf.x
        pred_cx = x[0] + x[3] * dt
        pred_cy = x[1] + x[4] * dt
        pred_cz = x[2] + x[5] * dt
        center = np.array([pred_cx, pred_cy, pred_cz])

        # 使用装甲板选择器获取未来时刻的最佳装甲板
        armor_id, armor_pos = armor_selector.select_armor(self, self_rel_pos, dt)
        return center, armor_pos


class MultiTargetTracker:
    def __init__(self, max_lost_time=0.5, match_threshold=0.4):
        self.targets = []
        self.max_lost_time = max_lost_time
        self.match_threshold = match_threshold

    def push_observation(self, obs_list, t):
        for target in self.targets:
            dt = t - target.last_update
            target.predict(dt)

        matches = {j: [] for j in range(len(self.targets))}
        for i, obs in enumerate(obs_list):
            best_target_idx = None
            min_dist = self.match_threshold
            for j, target in enumerate(self.targets):
                if target.robot_type != obs.robot_type:
                    continue
                dist = target.ekf.get_geometric_distance(obs.rel_pos)
                if dist < min_dist:
                    min_dist = dist
                    best_target_idx = j
            if best_target_idx is not None:
                matches[best_target_idx].append(i)

        used_obs_indices = set()
        for j, obs_indices in matches.items():
            target = self.targets[j]
            for i in obs_indices:
                obs = obs_list[i]
                target.update(obs, t)
                used_obs_indices.add(i)

        for j, tar in enumerate(self.targets):
            if j not in matches or not matches[j]:
                tar.on_missing()

        for i in range(len(obs_list)):
            if i not in used_obs_indices:
                self._create_target(obs_list[i], t)

        self.targets = [t for t in self.targets if t.state != 'lost']

    def _create_target(self, obs, t):
        new_target = TrackedTarget(obs, t)
        self.targets.append(new_target)

    def get_best_target(self):
        def state_priority(state):
            return {'tracking': 0, 'detecting': 1, 'temp_lost': 2, 'lost': 3}[state]

        active_targets = [t for t in self.targets if t.is_active()]
        if not active_targets:
            return None
        best = min(active_targets, key=lambda t: (state_priority(t.state), -t.update_count))
        return best



