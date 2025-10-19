import time

from object.abstract_object.simulator_rel.Shower import Shower
from object.abstract_object.simulator_rel.RobotManager import RobotManage

from object.real_object.Robot import RobotType


class Simulator:
    def __init__(self):
        self.t_last = time.time()

        self.shower = Shower()
        self.robot_manage = RobotManage()

    def run(self):
        self.update(self.robot_manage.robots)
        self.shower.show(self.robot_manage.robots)

    def update(self, robots):
        t_cur = time.time()
        dt = t_cur - self.t_last
        self.t_last = t_cur

    def ui(self):
        self.robot_manage.create_robot(RobotType.Hero)










