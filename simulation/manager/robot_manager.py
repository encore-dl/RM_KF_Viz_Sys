import numpy as np
import copy

from object.entity.robot import (Robot)


class RobotManage:
    def __init__(self, camera=None):
        self.robots = []
        self.obsrv_armors = []
        self.selected_robot = None
        self.camera = camera

        self.noise_sigma = 0.2

    def create_robot(self, robot_type):
        robot = Robot(robot_type=robot_type)
        self.robots.append(robot)

        self.selected_robot = self.robots[0]  # 暂定 先这样

    def delete_robot(self, robot_id):
        if robot_id >= len(self.robots) or robot_id < 0:
            print('robot_id is out of range!')
            return
        if self.selected_robot == self.robots[robot_id]:
            self.selected_robot = None
        self.robots.remove(self.robots[robot_id])

    def get_robots_count(self):
        return len(self.robots)

    def get_obsrv_armors(self, camera):  # 职能是给 robot的armor属性的位置属性 增添噪声，并输出该数据
        self.obsrv_armors = []

        for robot in self.robots:
            for i, armor in enumerate(robot.armors):
                noise = np.random.normal(0, self.noise_sigma, 3)
                obsrv_armor = copy.deepcopy(armor)
                obsrv_armor.world_pos += noise
                if camera.is_armor_visible(obsrv_armor.world_pos, robot.world_pos):
                    self.obsrv_armors.append(obsrv_armor)





