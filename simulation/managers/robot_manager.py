import numpy as np

from core.entities.rigid.robot import Robot


class RobotManager:
    def __init__(self, motion_manager):
        self.robots = []
        self.controlled_robot = None
        self.viewing_robot = None
        self.motion_manager = motion_manager  # 引用 MotionManager

    def create_robot(self, robot_type, chassis_pos=None, chassis_rpy=None,
                     gimbal_mount_pos=np.array([0., 0., 0.]), gimbal_mount_rpy=np.array([0., 0., 0.])):
        robot = Robot(robot_type, chassis_pos, chassis_rpy,
                      gimbal_mount_pos,
                      gimbal_mount_rpy)
        self.robots.append(robot)
        # 将底盘和云台添加到运动管理器
        self.motion_manager.add_entity(robot.chassis)
        self.motion_manager.add_entity(robot.gimbal)
        if self.controlled_robot is None:
            self.controlled_robot = robot
        if self.viewing_robot is None:
            self.viewing_robot = robot
        return robot

    def delete_robot(self, robot_id):
        if 0 <= robot_id < len(self.robots):
            robot = self.robots[robot_id]
            self.motion_manager.remove_entity(robot.chassis)
            self.motion_manager.remove_entity(robot.gimbal)
            if self.controlled_robot == robot:
                self.controlled_robot = None
            if self.viewing_robot == robot:
                self.viewing_robot = None
            self.robots.pop(robot_id)
            # 如果被删除后没有controlled_robot，选择第一个
            if self.robots and self.controlled_robot is None:
                self.controlled_robot = self.robots[0]
            if self.robots and self.viewing_robot is None:
                self.viewing_robot = self.robots[0]
        else:
            print(f"Robot index {robot_id} out of range")

    def switch_control_robot(self):
        if not self.robots:
            return None
        if self.controlled_robot is None:
            self.controlled_robot = self.robots[0]
        else:
            idx = self.robots.index(self.controlled_robot)
            idx = (idx + 1) % len(self.robots)
            self.controlled_robot = self.robots[idx]
        return self.controlled_robot
    
    def switch_view_robot(self):
        if not self.robots:
            return None
        if self.viewing_robot is None:
            self.viewing_robot = self.robots[0]
        else:
            idx = self.robots.index(self.viewing_robot)
            idx = (idx + 1) % len(self.robots)
            self.viewing_robot = self.robots[idx]
        return self.viewing_robot




