import time

import numpy as np

from object.model.tongji.tracking.track_state_machine import TrackStateMachine
from object.model.tongji.tracking.track_state_machine import MachineState
from object.model.tongji.tongji_model import TongJiModel
from object.entity.robot import Robot


class TongJiTracker:
    def __init__(self):
        self.tongji_model = TongJiModel()
        self.track_state_machine = TrackStateMachine()

        self.state = MachineState.lost

        self.tracked_robot = None

    def init_model(self, obsrv_armor=None):
        if obsrv_armor is None:
            return False

        self.tongji_model.init_model(obsrv_armor)
        self.tracked_robot = Robot(obsrv_armor.robot_type)

        return True

    def run_model(self, obsrv_armors, dt):
        if not self.tongji_model:
            return False

        self.tongji_model.predict(dt)

        found_count = 0
        min_x = float('inf')
        for armor in obsrv_armors:
            # if (obsrv_robot.robot_type != self.tongji_model.):
            if (armor.robot_type != self.tracked_robot.robot_type or
                    armor.armor_size != self.tracked_robot.armor_size):
                continue

            found_count += 1
            min_x = min(min_x, armor.world_pos[0])

        if found_count == 0:
            return False

        for armor in obsrv_armors:
            if (armor.robot_type != self.tracked_robot.robot_type or
                    armor.armor_size != self.tracked_robot.armor_size):
                continue

            self.tongji_model.update(armor)  # 本来是有个solve_pnp的，但是这里没，所以直接提供3d值
            break  # 暂定 先只更新一个

        return True

    def track(self, obsrv_armors, dt, camera_screen_center):
        if obsrv_armors is not None:
            obsrv_armors.sort(key=lambda a: np.linalg.norm(
                np.array([a.world_pos[0] * 1000, a.world_pos[1] * 1000]) - camera_screen_center
            ))

            obsrv_armors.sort(key=lambda a: a.priority)

        found = False
        if self.track_state_machine.state == MachineState.lost:
            found = self.init_model(obsrv_armors[0])
        else:
            found = self.run_model(obsrv_armors, dt)

        # 更新状态机
        self.state = self.track_state_machine.state_change(found, self.tracked_robot.robot_type)

        # 检测是否发散
        if self.state != MachineState.lost and self.tongji_model.diverged():
            print("model diverged!")
            self.state = MachineState.lost
            return []

        # 检查收敛状况
        if (self.state != MachineState.lost and  #
           self.tongji_model.get_ekf().data["recent_nis_failures"] >= 0.4 * self.tongji_model.get_ekf().window_size):
            print("bad convergence!")
            self.state = MachineState.lost
            return []

        if self.state == MachineState.lost:
            return []

        return  # 输出啥呢？




