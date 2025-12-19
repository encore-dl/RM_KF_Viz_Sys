from object.model.tongji.tracking.track_state_machine import TrackStateMachine
from object.model.tongji.tracking.track_state_machine import MachineState
from object.model.tongji.tongji_model import TongJiModel
from object.entity.robot import Robot


class TongJiTracker:
    def __init__(self):
        self.model = TongJiModel()

        self.tracked_robot = None
        self.is_tracked = False

        self.pred_pos = []  # 顺序是：0 车体中心， 1 2 3 4 装甲板中心

        self.track_state_machine = TrackStateMachine()
        self.state = MachineState.lost

    def track(self, obsrv_armors, dt, t_stamp):
        # if obsrv_armors is not None:
        #     # 图像中心排序
        #     obsrv_armors.sort(key=lambda a: np.linalg.norm(
        #         np.array([a.world_pos[0] * 1000, a.world_pos[1] * 1000]) - camera_screen_center
        #     ))
        #     # 击打优先级排序
        #     obsrv_armors.sort(key=lambda a: a.priority)

        self.is_tracked = False
        if not obsrv_armors or len(obsrv_armors) == 0:
            self.state = MachineState.lost
            return

        # 目标的跟踪过程
        found = False
        if self.track_state_machine.state == MachineState.lost:
            found = self.init_model(obsrv_armors[0])  # 用 obsrv_armors 的第一个来初始化
        else:
            found = self.run_model(obsrv_armors, dt)
        # 更新状态机
        self.state = self.track_state_machine.state_change(found, self.tracked_robot.robot_type)

        self.pred_pos = [self.model.get_pred_robot_pos(), *self.model.get_pred_armor_pos()]

        # self.is_tracked = True
        # return

        # 已经发散
        if self.state == MachineState.lost:
            self.is_tracked = False
            return

        # 检测是否发散
        if self.state != MachineState.lost and self.model.diverged():
            print("model diverged!")
            self.state = MachineState.lost
            self.is_tracked = False
            return

        # 检查收敛状况
        if self.state != MachineState.lost and self.model.nis_failed():
            print("bad convergence!")
            self.state = MachineState.lost
            self.is_tracked = False
            return

        if self.state == MachineState.tracking and not self.model.diverged():
            self.is_tracked = True
            return

    def init_model(self, obsrv_armor=None):
        if obsrv_armor is None:
            return False

        self.tracked_robot = Robot(obsrv_armor.robot_type)
        self.model.init_model(obsrv_armor, self.tracked_robot.armor_count)

        return True

    def run_model(self, obsrv_armors, dt):
        if not self.model:
            return False

        # kalman预测
        self.model.predict(dt)

        # 统计和 tracked robot 匹配的 obsrv 装甲板
        found_count = 0
        min_x = float('inf')
        for obsrv_armor in obsrv_armors:
            if (obsrv_armor.robot_type != self.tracked_robot.robot_type or
                    obsrv_armor.armor_size != self.tracked_robot.armor_size):
                continue

            found_count += 1
            min_x = min(min_x, obsrv_armor.world_pos[0])

        if found_count == 0:
            return False

        # kalman更新
        # n = 4
        for obsrv_armor in obsrv_armors:
            if (obsrv_armor.robot_type != self.tracked_robot.robot_type or
                    obsrv_armor.armor_size != self.tracked_robot.armor_size):
                continue

            self.model.update(obsrv_armor)  # 本来是有个solve_pnp的，但是这里没，所以直接提供3d值

            # n -= 1
            # if n == 0:
            #     break

        return True






