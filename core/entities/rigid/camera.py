import numpy as np
import math

from core.entities.rigid.rigid import Rigid
from core.algorithms.math import euler_to_rotation_matrix


class Camera(Rigid):
    def __init__(self, fov=60, max_range=10, **kwargs):
        super().__init__(**kwargs)

        self.fov = math.radians(fov)  # field of view 视场角 弧度制
        self.max_range = max_range  # 相机最远识别范围/距离
        self.focal_len = 800  # 焦距 单位：像素

        self.R_body_to_optical = np.array([
            [0, -1, 0],  # Optical X axis comes from -Body Y
            [0, 0, -1],  # Optical Y axis comes from -Body Z
            [1, 0, 0]  # Optical Z axis comes from +Body X
        ])

        self.auto_aiming = False

    def world_to_camera(self, world_pos):
        # 这里是固定相机在世界坐标系原点，让所有点平移过去，然后相机不转，点转
        # 但是相机的旋转和相机视角里点的旋转是相反的，作用在点上的旋转矩阵应当是R的逆
        # 但R是正交矩阵，所以R转置等于R的逆
        rel_world_pos = world_pos - self.world_pos
        R = euler_to_rotation_matrix(self.world_rpy)
        R_total = self.R_body_to_optical @ R.T

        optical_pos = R_total @ rel_world_pos
        if optical_pos[2] <= 0:
            return None
        
        return optical_pos

    def camera_to_pixel(self, optical_pos):
        """
        标准针孔相机模型 projection
        Input: optical_pos [x(right), y(down), z(forward)]
        """

        if optical_pos is None:
            return None
        
        x, y, z = optical_pos

        u_norm = x / z 
        v_norm = y / z  

        u = u_norm * self.focal_len
        v = v_norm * self.focal_len

        # return np.array([int(u), int(v)])
        # return np.array([round(u, 1), round(v, 1)])
        # return np.array([round(u, 1), round(v, 2)])
        return np.array([u, v])

    def world_to_pixel(self, world_pos):
        optical_pos = self.world_to_camera(world_pos)
        if optical_pos is not None and self.is_in_fov(optical_pos):
            return self.camera_to_pixel(optical_pos)
        return None

    def is_in_fov(self, optical_pos):
        distance = np.linalg.norm(optical_pos)
        if distance > self.max_range:  # 点过远 或 在相机后面
            return False

        # 圆锥型视角 判断是否超出
        azimuth = math.atan2(optical_pos[0], optical_pos[2])
        elevation = math.atan2(optical_pos[1], optical_pos[2])

        return abs(azimuth) <= self.fov / 2 and abs(elevation) <= self.fov / 2

    def look_at(self, aiming_pos):
        r_vec = aiming_pos - self.world_pos
        dx, dy, dz = r_vec
        dist_xy = math.sqrt(dx ** 2 + dy ** 2)

        self.world_rpy[2] = math.atan2(dy, dx)
        self.world_rpy[1] = np.clip(-math.atan2(dz, dist_xy), -math.pi / 2, math.pi / 2)

    def get_forward_vec(self):
        R = euler_to_rotation_matrix(self.world_rpy)
        forward_vec = R @ np.array([0.01, 0., 0.])

        return forward_vec

    def is_armor_visible(self, armor_world_pos, robot_world_pos):
        armor_to_camera = self.world_pos - armor_world_pos
        robot_to_armor = armor_world_pos - robot_world_pos

        norm_a = np.linalg.norm(armor_to_camera)
        norm_r = np.linalg.norm(robot_to_armor)
        if norm_a == 0 or norm_r == 0:
            return False
        cos_angle = np.dot(robot_to_armor, armor_to_camera) / (norm_r * norm_a)

        return cos_angle > np.sqrt(1)/2

    def get_intrinsic_matrix(self):
        return np.array([
            [self.focal_len, 0, 0],
            [0, self.focal_len, 0],
            [0, 0, 1]
        ])

    def switch_auto_aiming(self):
        self.auto_aiming = not self.auto_aiming

    def apply_control(self, alpha, dt):
        self.world_alp[2] = alpha
        self.world_omg[2] += self.world_alp[2] * dt
        self.world_rpy[2] += self.world_omg[2] * dt

