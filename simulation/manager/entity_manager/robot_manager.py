import math

import numpy as np
import copy
import time

from collections import deque

from object.entity.robot import (Robot)
from utils.math_tool import safe_angle_sub, rad_to_ratio


class RobotManager:
    def __init__(self, camera=None):
        self.robots = []
        self.obsrv_data_with_t = None
        self.selected_robot = None
        self.camera = camera

        self.noise_sigma = 0.000

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

    def get_obsrv(self, camera):  # 职能是给 robot的armor属性的位置属性 增添噪声，并输出该数据
        obsrv_armors = []

        for robot in self.robots:
            for i, armor in enumerate(robot.armors):
                noise = np.random.normal(0, self.noise_sigma, 3)
                obsrv_armor = copy.deepcopy(armor)
                obsrv_armor.world_pos += noise
                if camera.is_armor_visible(obsrv_armor.world_pos, robot.world_pos):
                    obsrv_armors.append(obsrv_armor)

        self.obsrv_data_with_t = (
            obsrv_armors,
            time.time()
        )





