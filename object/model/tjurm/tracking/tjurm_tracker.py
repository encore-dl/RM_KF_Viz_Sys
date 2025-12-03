# from object.model.tjurm.tjurm_model import TJURMModel
from object.model.tjurm.submodel.antitop import Antitop
from object.model.tongji.tracking.track_state_machine import TrackStateMachine, MachineState
from object.entity.robot import Robot


class TJURMTracker:
    """TJURM跟踪器"""

    def __init__(self):
        self.model = Antitop()
        self.track_state_machine = TrackStateMachine()

        self.state = MachineState.lost
        self.tracked_robot = None
        self.is_tracked = False

    def init_model(self, obsrv_armor=None):
        """初始化模型"""
        if obsrv_armor is None:
            return False

        self.tracked_robot = Robot(obsrv_armor.robot_type)
        self.model.init_model(obsrv_armor, self.tracked_robot.armor_count)

        return True

    def run_model(self, obsrv_armors, dt):
        """运行模型"""
        if not self.model:
            return False

        # 预测步骤
        self.model.predict(dt)

        # 统计匹配的观测装甲板
        found_count = 0
        for armor in obsrv_armors:
            if (armor.robot_type != self.tracked_robot.robot_type or
                    armor.armor_size != self.tracked_robot.armor_size):
                continue

            found_count += 1

        if found_count == 0:
            return False

        # 更新步骤
        for armor in obsrv_armors:
            if (armor.robot_type != self.tracked_robot.robot_type or
                    armor.armor_size != self.tracked_robot.armor_size):
                continue

            self.model.update(armor)
            break  # 先用一个装甲板更新

        return True

    def track(self, obsrv_armors, dt):
        """主跟踪函数"""
        self.is_tracked = False

        if not obsrv_armors or len(obsrv_armors) == 0:
            self.state = MachineState.lost
            return

        # 目标跟踪过程
        found = False
        if self.track_state_machine.state == MachineState.lost:
            found = self.init_model(obsrv_armors[0])
        else:
            found = self.run_model(obsrv_armors, dt)

        # 更新状态机
        self.state = self.track_state_machine.state_change(
            found,
            self.tracked_robot.robot_type if self.tracked_robot else None
        )

        if self.state == MachineState.lost:
            self.is_tracked = False
            return

        self.is_tracked = True



