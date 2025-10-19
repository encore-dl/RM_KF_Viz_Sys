from object.real_object.Robot import Robot, RobotType


class RobotManage:
    def __init__(self):
        self.robots = []

    def create_robot(self, robot_type):
        robot = Robot(robot_type=robot_type)
        self.robots.append(robot)

    def delete_robot(self, robot_id):
        if robot_id >= len(self.robots) or robot_id < 0:
            print('robot_id is out of range!')
            return
        self.robots.remove(self.robots[robot_id])

    def get_robots_count(self):
        return len(self.robots)

    def update_robots(self):
        pass






