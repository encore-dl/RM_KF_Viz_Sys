from enum import IntEnum


# 独立出来，防止import循环
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
        names = {
            cls.Hero: "Hero",
            cls.Sentry: "Sentry",
            cls.Infantry: "Infantry",
            cls.Engineer: "Engineer",
            cls.Outpost: "Outpost",
            cls.Base: "Base"
        }
        return names.get(robot_type, "")

