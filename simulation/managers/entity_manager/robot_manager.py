from core.entities.rigid.robot import (Robot)


class RobotManager:
    def __init__(self, camera=None):
        self.robots = []
        self.selected_robot = None
        self.camera = camera

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







