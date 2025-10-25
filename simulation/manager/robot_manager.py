import numpy as np

from object.entity.robot import (Robot)


class RobotManage:
    def __init__(self, camera=None):
        self.robots = []
        self.obsrv_armors = []
        self.selected_robot = None
        self.camera = camera

        self.noise_sigma = 0.1

    def create_robot(self, robot_type):
        robot = Robot(robot_type=robot_type)
        self.robots.append(robot)

        self.selected_robot = self.robots[0]  # 暂定 先这样

    def delete_robot(self, robot_id):
        if robot_id >= len(self.robots) or robot_id < 0:
            print('robot_id is out of range!')
            return
        self.robots.remove(self.robots[robot_id])

        self.selected_robot = self.robots[0]  # 暂定 先这样

    def get_robots_count(self):
        return len(self.robots)

    def get_obsrv_armors(self):  # 职能是给 robot的armor属性的位置属性 增添噪声，并输出该数据
        self.obsrv_armors = []

        noise = np.random.normal(0, self.noise_sigma, 3)
        for robot in self.robots:
            for i, armor in enumerate(robot.armors):
                obsrv_armor = armor
                obsrv_armor.world_pos += noise
                self.obsrv_armors.append(obsrv_armor)

    def update_robots(self):
        pass






