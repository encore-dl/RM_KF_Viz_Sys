import numpy as np
import math

from utils.math_tool import get_euler_rotate_matrix


class Camera:
    def __init__(self, resolution, world_pos=np.array([0., 0., 0.]), fov=60, max_range=10, orient=np.array([0., 0., 0.])):
        self.world_pos = world_pos
        self.world_vel = np.array([0., 0., 0.])
        self.world_rpy = orient

        self.fov = math.radians(fov)  # field of view 视场角
        self.max_range = max_range  # 相机最远识别范围/距离

        self.focal_len = 800  # 焦距 单位：像素
        self.resolution = resolution

        self.auto_aiming = False

    def world_to_camera(self, world_pos):
        rel_pos = world_pos - self.world_pos
        R = get_euler_rotate_matrix(self.world_rpy)
        camera_pos_temp = R @ rel_pos
        # 世界坐标系：x 右 y 前 z 上
        # 相机坐标系：x 右 y 下 z 前
        camera_pos = np.array([
            camera_pos_temp[0],
            -camera_pos_temp[2],
            camera_pos_temp[1]
        ])

        return camera_pos

    def camera_to_pixel(self, camera_pos):
        if camera_pos[2] < 0:
            return None

        x_norm = camera_pos[0] / camera_pos[2]  # x / z
        y_norm = camera_pos[1] / camera_pos[2]  # y / z

        u = x_norm * self.focal_len + self.resolution[0] / 2
        v = y_norm * self.focal_len + self.resolution[1] / 2

        if 0 <= u < self.resolution[0] and 0 <= v < self.resolution[1]:
            return int(u), int(v)
        else:
            return None

    def world_to_pixel(self, world_pos):
        camera_pos = self.world_to_camera(world_pos)
        return self.camera_to_pixel(camera_pos)

    def is_in_fov(self, world_pos):
        camera_pos = self.world_to_camera(world_pos)

        distance = np.linalg.norm(camera_pos)
        if distance > self.max_range or camera_pos[2] <= 0:  # 点过远
            return False

        # 圆锥型视角 判断是否超出
        horizontal_angle = math.atan2(camera_pos[0], camera_pos[2])
        vertical_angle = math.atan2(camera_pos[1], camera_pos[2])

        return abs(horizontal_angle) <= self.fov / 2 and abs(vertical_angle) <= self.fov / 2

    def look_at(self, aiming_pos):
        direction = aiming_pos - self.world_pos

        self.world_rpy[2] = math.atan2(direction[1], direction[0])

        hrz_dist = math.sqrt(direction[0] ** 2 + direction[1] ** 2)
        self.world_rpy[1] = math.atan2(-direction[2], hrz_dist)

        self.world_rpy[1] = np.clip(self.world_rpy[1], -math.pi / 2, math.pi / 2)

    # def














