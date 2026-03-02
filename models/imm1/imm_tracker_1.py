import numpy as np

from models.imm1.imm_model_1 import IMMModel1


class IMMTracker1:
    def __init__(self):
        self.model = IMMModel1()

        self.is_tracking = False
        self.pred_pos = None
        # 用于存储状态向量供外部调试/可视化 [motion_vec(8), rot_vec(6)]
        self.state_vecs = None

        self.last_t = 0.
        self.lost_t = 0.
        self.max_lost_t = 0.5
        self.match_thresh = 0.6

    def _reset_tracker(self):
        self.is_tracking = False
        self.lost_t = 0.

    def track(self, obs_armors, dt, timestamp):
        if not obs_armors:
            self.is_tracking = False
            self.pred_pos = []
            return

        # 1. 预测
        # IMM 模型内部会处理交互(Interaction)和各自的预测
        if self.is_tracking:
            self.model.predict(dt)
            self.lost_t += dt

        # 2. 匹配
        tar_armor = self._match_armor(obs_armors)

        if tar_armor is not None:
            self.lost_t = 0.

            # 3. 初始化 或 更新
            if not self.is_tracking:
                self.model.init_model(
                    tar_armor.world_pos,
                    tar_armor.world_rpy[2],
                    timestamp
                )
                self.is_tracking = True
            else:
                # IMM 模型内部会处理各自的更新、似然计算、概率更新和状态融合
                self.model.update(
                    tar_armor.world_pos,
                    tar_armor.world_rpy[2],
                    timestamp
                )

            self.last_t = timestamp

        else:
            # 丢失处理
            if self.is_tracking:
                if self.lost_t > self.max_lost_t:
                    self._reset_tracker()
                    print(f"DemoTracker: Lost at {timestamp:.3f}")

        # 4. 获取结果与状态拆分
        if self.is_tracking:
            # 获取击打预测点 (使用融合状态)
            self.pred_pos = self.get_pred_pos(0.1)

            # --- 关键修改：从 IMM 的 14维 fused_x 中拆分状态 ---
            fx = self.model.fused_x

            # 0-7: [x, y, z, vx, vy, vz, ax, ay]
            motion_vec = fx[:8].copy()

            # 8-13: [psi, w, alpha, ra, rb, dz]
            # 新模型中 alpha 已经在 index 10 了，直接切片即可
            rot_vec = fx[8:].copy()

            self.state_vecs = [motion_vec, rot_vec]

            # 调试打印 (可选): 查看当前 IMM 认为哪个模型更准
            # print(f"IMM Prob - Motion: {self.model.mu[0]:.2f}, Spin: {self.model.mu[1]:.2f}")

    def _match_armor(self, obsrv_armors):
        if not obsrv_armors or len(obsrv_armors) == 0:
            return None

        if not self.is_tracking:
            # 没有追踪时，默认取第一个
            return obsrv_armors[0]
        else:
            # 获取预测的车辆中心 (flight_time=0)
            # 这里的 get_pred_pos 已经是基于融合状态的了
            pred_center, _ = self.model.get_pred_pos(0.)

            best_armor = None
            min_dist = float('inf')

            for obsrv_armor in obsrv_armors:
                # 简单的最近邻匹配
                dist = np.linalg.norm(obsrv_armor.world_pos - pred_center)

                if dist < min_dist and dist < self.match_thresh:
                    min_dist = dist
                    best_armor = obsrv_armor

            return best_armor

    def get_pred_pos(self, fly_t):
        if not self.is_tracking:
            return np.zeros(3), np.zeros(3)

        # 直接调用 Model 的接口，Model 会用 fused_x 计算
        return self.model.get_pred_pos(fly_t)