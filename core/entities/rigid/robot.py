import numpy as np
from core.entities.rigid.chassis import Chassis
from core.entities.rigid.gimbal import Gimbal
from core.algorithms.math.transform import euler_to_rotation_matrix, world_to_robot


class Robot:
    """完整的机器人，包含底盘和云台"""
    def __init__(self, robot_type, chassis_pos=None, chassis_rpy=None,
                 gimbal_mount_pos=np.array([0, 0, 0.3]), gimbal_mount_rpy=np.zeros(3)):
        self.robot_type = robot_type
        # 底盘
        self.chassis = Chassis(robot_type, world_pos=chassis_pos, world_rpy=chassis_rpy)
        # 云台（安装在底盘上）
        self.gimbal = Gimbal(mount_pos=gimbal_mount_pos, mount_rpy=gimbal_mount_rpy)
        # 建立引用
        self.gimbal.owner_chassis = self.chassis
        self.gimbal.world_rpy = self.chassis.world_rpy + gimbal_mount_rpy  # 简化叠加
        self.gimbal.world_pos = self.chassis.world_pos + euler_to_rotation_matrix(self.chassis.world_rpy) @ gimbal_mount_pos

        # 初始化云台世界位姿
        self.gimbal.update_from_chassis(self.chassis)

    def get_armors(self):
        return self.chassis.armors

    def get_camera(self):
        return self.gimbal.camera

    def get_muzzle(self):
        return self.gimbal.muzzle

    def get_camera_rel_pos(self):
        return world_to_robot(self.get_camera().world_pos, self.chassis)

    def get_muzzle_rel_pos(self):
        return world_to_robot(self.get_muzzle().world_pos, self.chassis)

    def get_gimbal_rel_pos(self):
        return world_to_robot(self.gimbal.world_pos, self.chassis)

