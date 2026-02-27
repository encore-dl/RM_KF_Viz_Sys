import numpy as np
import math
from core.entities.rigid.rigid import Rigid
from core.algorithms.math import euler_to_rotation_matrix


class Camera(Rigid):
    def __init__(self, mount_pos=np.zeros(3), mount_rpy=np.zeros(3),
                 fov=60, max_range=10, **kwargs):
        super().__init__(**kwargs)
        self.mount_pos = mount_pos.copy()   # 相对于云台的安装位置
        self.mount_rpy = mount_rpy.copy()   # 相对于云台的安装姿态
        self.fov = math.radians(fov)
        self.max_range = max_range
        self.focal_len = 800
        self.R_body_to_optical = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ])

    def world_to_camera(self, world_pos):
        rel_world_pos = world_pos - self.world_pos
        R = euler_to_rotation_matrix(self.world_rpy)
        R_total = self.R_body_to_optical @ R.T
        optical_pos = R_total @ rel_world_pos
        if optical_pos[2] <= 0:
            return None
        return optical_pos

    def camera_to_pixel(self, optical_pos):
        if optical_pos is None:
            return None
        x, y, z = optical_pos
        u_norm = x / z
        v_norm = y / z
        u = u_norm * self.focal_len
        v = v_norm * self.focal_len
        return np.array([u, v])

    def world_to_pixel(self, world_pos):
        optical_pos = self.world_to_camera(world_pos)
        if optical_pos is not None and self.is_in_fov(optical_pos):
            return self.camera_to_pixel(optical_pos)
        return None

    def is_in_fov(self, optical_pos):
        distance = np.linalg.norm(optical_pos)
        if distance > self.max_range:
            return False
        azimuth = math.atan2(optical_pos[0], optical_pos[2])
        elevation = math.atan2(optical_pos[1], optical_pos[2])
        return abs(azimuth) <= self.fov / 2 and abs(elevation) <= self.fov / 2

    def get_forward_vec(self):
        R = euler_to_rotation_matrix(self.world_rpy)
        return R @ np.array([1.0, 0, 0])

    def is_armor_visible(self, armor_world_pos, robot_world_pos):
        armor_to_camera = self.world_pos - armor_world_pos
        robot_to_armor = armor_world_pos - robot_world_pos
        norm_a = np.linalg.norm(armor_to_camera)
        norm_r = np.linalg.norm(robot_to_armor)
        if norm_a == 0 or norm_r == 0:
            return False
        cos_angle = np.dot(robot_to_armor, armor_to_camera) / (norm_r * norm_a)
        return cos_angle > 0.5

    def get_intrinsic_matrix(self):
        return np.array([
            [self.focal_len, 0, 0],
            [0, self.focal_len, 0],
            [0, 0, 1]
        ])


