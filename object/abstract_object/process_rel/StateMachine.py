from enum import Enum

from object.real_object.Robot import RobotType


class MachineState(Enum):
    lost = 1
    temporary_lost = 2
    detecting = 3
    tracking = 4
    switching = 5


class StateMachine:
    def __init__(self, target):
        self.state = MachineState.lost
        self.pre_state = MachineState.lost

        self.detect_count = 0
        self.temp_lost_count = 0

        self.min_detect_count = 5
        self.max_temp_lost_count = 15
        self.normal_temp_lost_count = 15
        self.outpost_max_temp_lost_count = 75

        self.target = target

    def state_change(self, found):
        if self.state == MachineState.lost:
            if not found:
                return
            self.state = MachineState.detecting
            self.detect_count = 1

        elif self.state == MachineState.detecting:
            if found:
                self.detect_count += 1
                if self.detect_count >= self.min_detect_count:
                    self.state = MachineState.tracking
            else:
                self.detect_count = 0
                self.state = MachineState.lost

        elif self.state == MachineState.tracking:
            if not found:
                self.temp_lost_count = 1
                self.state = MachineState.temporary_lost

        elif self.state == MachineState.switching:
            if found:
                self.state = MachineState.detecting
            else:
                self.temp_lost_count += 1
                if self.temp_lost_count > 200:
                    self.state = MachineState.lost

        elif self.state == MachineState.temporary_lost:
            if found:
                self.state = MachineState.tracking
            else:
                self.temp_lost_count += 1

                if self.target.name == RobotType.Outpost:
                    self.max_temp_lost_count = self.outpost_max_temp_lost_count
                else:
                    self.max_temp_lost_count = self.normal_temp_lost_count

                if self.temp_lost_count > self.max_temp_lost_count:
                    self.state = MachineState.lost








