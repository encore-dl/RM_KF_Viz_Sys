import numpy as np

from object.model.demo.demo_model import DemoModel


class DemoTracker:
    def __init__(self):
        self.model = DemoModel()

        self.is_tracking = False
        self.pred_pos = None
        self.state_vecs = None

        self.last_t = 0.
        self.lost_t = 0.
        self.max_lost_t = 0.5
        self.match_thresh = 0.6

    def _reset_tracker(self):
        self.is_tracking = False
        self.lost_t = 0.

    def track(self, obsrv_armors, dt, t_stamp):
        if self.is_tracking:
            self.model.predict(dt)
            self.lost_t += dt

        tar_armor = self._match_armor(obsrv_armors)

        if tar_armor is not None:
            self.lost_t = 0.

            if not self.is_tracking:
                self.model.init_model(
                    tar_armor.world_pos,
                    tar_armor.world_rpy[2],
                    t_stamp
                )
                self.is_tracking = True
            else:
                self.model.update(
                    tar_armor.world_pos,
                    tar_armor.world_rpy[2],
                    t_stamp
                )

            self.last_t = t_stamp

        else:
            if self.is_tracking:
                if self.lost_t > self.max_lost_t:
                    self._reset_tracker()
                    print(f"DemoTracker: Lost at {t_stamp:.3f}")

        if self.is_tracking:
            self.pred_pos = self.get_pred_pos(0.1)
            self.state_vecs = [
                self.model.kf_motion.x,
                self.model.ukf_rot.x
            ]

    def _match_armor(self, obsrv_armors):
        if not obsrv_armors or len(obsrv_armors) == 0:
            return None

        if not self.is_tracking:
            return obsrv_armors[0]
        else:
            pred_center, _ = self.model.get_pred_pos(0.)

            best_armor = None
            min_dist = float('inf')

            for obsrv_armor in obsrv_armors:
                dist = np.linalg.norm(obsrv_armor.world_pos - pred_center)

                if dist < min_dist and dist < self.match_thresh:
                    min_dist = dist
                    best_armor = obsrv_armor

            return best_armor

    def get_pred_pos(self, fly_t):
        if not self.is_tracking:
            return np.zeros(3), np.zeros(3)

        return self.model.get_pred_pos(fly_t)









