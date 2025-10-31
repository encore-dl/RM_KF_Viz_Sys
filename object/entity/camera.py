import numpy as np
import math

from utils.math_tool import get_euler_rotate_matrix, pos_to_tpd


class Camera:
    def __init__(self, world_pos=np.array([0., 0., 0.]), fov=60, max_range=10, orient=np.array([0., 0., 0.])):
        self.world_pos = world_pos
        self.world_vel = np.array([0., 0., 0.])
        self.world_tpd = pos_to_tpd(world_pos)
        self.world_rpy = orient
        self.world_omg = np.array([0., 0., 0.])

        self.fov = math.radians(fov)  # field of view 视场角 弧度制
        self.max_range = max_range  # 相机最远识别范围/距离
        self.focal_len = 800  # 焦距 单位：像素

        self.auto_aiming = False

    def world_to_pixel(self, world_pos, camera_screen_center, resolution):
        camera_pos = self.world_to_camera(world_pos)
        return self.camera_to_pixel(camera_pos, camera_screen_center, resolution)

    def world_to_camera(self, world_pos):
        # 这里是固定相机在世界坐标系原点，让所有点平移过去，然后相机不转，点转
        # 但是相机的旋转和相机视角里点的旋转是相反的，作用在点上的旋转矩阵应当是R的逆
        # 但R是正交矩阵，所以R转置等于R的逆
        rel_pos = world_pos - self.world_pos
        R = get_euler_rotate_matrix(self.world_rpy)
        camera_pos_temp = R.T @ rel_pos
        # 世界坐标系：x前 y右 z上
        # 相机坐标系：x右 y下 z前
        camera_pos = np.array([
            camera_pos_temp[1],  # y右 -> x右
            -camera_pos_temp[2],  # z上 -> y下
            camera_pos_temp[0]   # x前 -> z前
        ])

        if self.is_in_fov(camera_pos):
            return camera_pos
        else:
            return None

    def camera_to_pixel(self, camera_pos, camera_screen_center, resolution):
        if camera_pos is None or camera_pos[2] < 0:
            return None

        x_norm = camera_pos[0] / camera_pos[2]  # x / z
        y_norm = camera_pos[1] / camera_pos[2]  # y / z

        u = x_norm * self.focal_len
        v = y_norm * self.focal_len

        if -resolution[0]/2 <= u < resolution[0]/2 and -resolution[1]/2 <= v < resolution[1]/2:
            u += camera_screen_center[0]
            v += camera_screen_center[1]

            return np.array([int(u), int(v)])
        else:
            return None

    def is_in_fov(self, camera_pos):
        distance = np.linalg.norm(camera_pos)
        if distance > self.max_range or camera_pos[2] <= 0:  # 点过远 或 在相机后面
            return False

        # 圆锥型视角 判断是否超出
        horizontal_angle = math.atan2(camera_pos[0], camera_pos[2])
        vertical_angle = math.atan2(camera_pos[1], camera_pos[2])

        return abs(horizontal_angle) <= self.fov / 2 and abs(vertical_angle) <= self.fov / 2

    def look_at(self, aiming_pos):
        direction = aiming_pos - self.world_pos

        self.world_rpy[2] = math.atan2(direction[1], direction[0])

        xy_dist = math.sqrt(direction[0] ** 2 + direction[1] ** 2)
        self.world_rpy[1] = math.atan2(-direction[2], xy_dist)
        self.world_rpy[1] = np.clip(self.world_rpy[1], -math.pi / 2, math.pi / 2)

    def get_forward_vec(self):
        R = get_euler_rotate_matrix(self.world_rpy)
        forward_vec = R @ np.array([1., 0., 0.])

        return forward_vec

    def is_armor_visible(self, armor_world_pos, robot_world_pos):
        robot_camera_vec = self.world_pos - robot_world_pos
        robot_camera_univec = robot_camera_vec / np.linalg.norm(robot_camera_vec)

        robot_armor_vec = armor_world_pos - robot_world_pos
        robot_armor_univec = robot_armor_vec / np.linalg.norm(robot_armor_vec)

        dot_product = np.dot(robot_camera_univec, robot_armor_univec)

        return 0.5 <= dot_product <= 1  # 内积在 [√3/2, 1] 之间算看见
