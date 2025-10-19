import time

from simulation.visualizer import Visualizer
from simulation.managers.robot_manager import RobotManage

from object.entity.Robot import RobotType


class Simulator:
    def __init__(self):
        self.t_last = time.time()

        self.shower = Visualizer()
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










