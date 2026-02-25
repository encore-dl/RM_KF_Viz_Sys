from core.entities.rigid.robot import Robot


class RobotManager:
    def __init__(self):
        self.robots = []
        self.selected_robot = None

    def create_robot(self, robot_type):
        robot = Robot(robot_type=robot_type)
        self.robots.append(robot)
        if self.selected_robot is None:
            self.selected_robot = robot

    def delete_robot(self, robot_id):
        if 0 <= robot_id < len(self.robots):
            if self.selected_robot == self.robots[robot_id]:
                self.selected_robot = None
            self.robots.pop(robot_id)
            if self.robots and self.selected_robot is None:
                self.selected_robot = self.robots[0]
        else:
            print(f"Robot index {robot_id} out of range")

    def next_robot(self):
        if not self.robots:
            return None
        if self.selected_robot is None:
            self.selected_robot = self.robots[0]
        else:
            idx = self.robots.index(self.selected_robot)
            idx = (idx + 1) % len(self.robots)
            self.selected_robot = self.robots[idx]
        return self.selected_robot







