import numpy as np

from object.entity.robot import (Robot)


class RobotManage:
    def __init__(self, camera=None):
        self.robots = []
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

    def generate_observed_robots(self):
        observed_robots = []

        noise = np.random.normal(0, self.noise_sigma, 3)
        for robot in self.robots:
            observed_robot = robot
            for i, armor in enumerate(robot.armors):
                observed_armor = armor
                observed_armor.world_pos += noise
                observed_robot.armors[i] = observed_armor

            observed_robot.world_pos += noise

            observed_robots.append(observed_robot)

        return observed_robots

    def update_robots(self):
        pass






