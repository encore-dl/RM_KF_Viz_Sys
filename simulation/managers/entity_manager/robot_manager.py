import numpy as np

from core.entities.rigid.robot import Robot


class RobotManager:
    def __init__(self, motion_manager):
        self.robots = []
        self.selected_robot = None
        self.motion_manager = motion_manager  # 引用 MotionManager

    def create_robot(self, robot_type, chassis_pos=None, chassis_rpy=None,
                     gimbal_mount_pos=None, gimbal_mount_rpy=None):
        robot = Robot(robot_type, chassis_pos, chassis_rpy,
                      gimbal_mount_pos or np.array([0, 0, 0.3]),
                      gimbal_mount_rpy or np.zeros(3))
        self.robots.append(robot)
        # 将底盘和云台添加到运动管理器
        self.motion_manager.add_entity(robot.chassis)
        self.motion_manager.add_entity(robot.gimbal)
        if self.selected_robot is None:
            self.selected_robot = robot
        return robot

    def delete_robot(self, robot_id):
        if 0 <= robot_id < len(self.robots):
            robot = self.robots[robot_id]
            # 从运动管理器中移除
            self.motion_manager.remove_entity(robot.chassis)
            self.motion_manager.remove_entity(robot.gimbal)
            if self.selected_robot == robot:
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