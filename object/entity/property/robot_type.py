from enum import IntEnum


# 车的型号，顺便排列优先级
class RobotType(IntEnum):
    Hero = 1
    Sentry = 2
    Infantry = 3
    Engineer = 4
    Outpost = 5
    Base = 6

    @classmethod
    def get_name(cls, robot_type):
        robot_name_str = ''
        if robot_type == RobotType.Hero:
            robot_name_str = 'Hero'
        elif robot_type == RobotType.Sentry:
            robot_name_str = 'Sentry'
        elif robot_type == RobotType.Infantry:
            robot_name_str = 'Infantry'
        elif robot_type == RobotType.Engineer:
            robot_name_str = 'Engineer'
        return robot_name_str

