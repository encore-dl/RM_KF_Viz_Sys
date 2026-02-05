import numpy as np
import copy
import time

from object.entity.rigid.camera import Camera
from utils.math_tool import euler_to_rotation_matrix, limit_rad
from algorithm.perspective_n_point.perspective_n_point import solve_pnp_core


class CameraManager:
    def __init__(self):
        self.cameras = []
        self.selected_camera = None

        self.camera = Camera(
            world_pos=np.array([0., 0., 0.15]),
            world_rpy=np.array([0., 0., 0.]),
            fov=360,
            max_range=10,
        )

        self.obsrv_data_with_t = None

        self.pos_noise_sigma = 0.00
        self.rpy_noise_sigma = 0.0

        self.pnp_result_info = None

    def switch_auto_aiming(self):
        self.camera.auto_aiming = not self.camera.auto_aiming

    def get_obsrv(self, robots):  # 职能是给 robot的armor属性的位置属性 增添噪声，并输出该数据
        obsrv_armors = []

        for robot in robots:
            for armor in robot.armors:
                if not self.camera.is_armor_visible(armor.world_pos, robot.world_pos):
                    continue

                # armor.priority = self.camera.world_to_pixel(armor.world_pos)

                if armor.armor_size == 'large':
                    w, h = armor.light_bar_interval, armor.light_bar_length
                else:  # small
                    w, h = armor.light_bar_interval, armor.light_bar_length

                # 装甲板局部坐标系下的4个角点（FLU坐标系）
                local_corners = np.array([
                    [0, w / 2, h / 2],  # 左上
                    [0, -w / 2, h / 2],  # 右上
                    [0, -w / 2, -h / 2],  # 右下
                    [0, w / 2, -h / 2]  # 左下
                ])

                # 将局部坐标变换到世界坐标系
                # 旋转矩阵：从欧拉角计算
                R_armor = euler_to_rotation_matrix(armor.world_rpy)

                # 变换：R * local + T
                # target_armor.world_pos 是带噪声的观测位置
                corners_3d_world = (R_armor @ local_corners.T).T + armor.world_pos

                # 3. 投影到像素坐标并添加噪声
                pixel_points = []
                for corner_3d in corners_3d_world:
                    # 投影到像素坐标
                    uv = self.camera.world_to_pixel(corner_3d)
                    if uv is None:
                        self.pnp_result_info = None  # 超出视野
                        return

                    # 添加像素噪声（模拟检测误差）
                    # 标准差0.5像素
                    noise_uv = np.random.normal(0, 0.)
                    noisy_uv = np.array([uv[0] + noise_uv, uv[1] + noise_uv], dtype=np.float32)
                    pixel_points.append(noisy_uv)

                ordered_pixels = np.array(pixel_points, dtype=np.float32)

                # 4. 准备PnP求解参数
                K = self.get_intrinsic_matrix()  # 相机内参
                D = np.zeros(5)  # 畸变系数（模拟器假设无畸变）

                # 4.1 获取相机（云台）在世界坐标系中的位姿
                R_head2world = euler_to_rotation_matrix(self.camera.world_rpy)  # 相机旋转矩阵
                T_head2world = self.camera.world_pos  # 相机位置

                # 4.2 PnP坐标系到相机坐标系的变换
                # 假设PnP求解结果在相机坐标系（RDF）中
                # R_body_to_optical是从车身（FLU）到相机（RDF）的旋转
                R_pnp2head = self.camera.R_body_to_optical.T  # 相机到车身的旋转（逆）
                T_pnp2head = np.zeros(3)  # 平移设为0（简化）

                # 4.3 当前云台偏航角
                head_yaw = self.camera.world_rpy[2]

                # 5. 调用PnP求解器
                calc_pos, calc_rpy = solve_pnp_core(
                    camera_k=K,  # 相机内参
                    camera_dist=D,  # 畸变系数
                    armor_type='large',  # 装甲板类型
                    image_points_2d=ordered_pixels,  # 2D像素点
                    head_yaw=head_yaw,  # 云台当前偏航角
                    trans_head2world=T_head2world,  # 云台到世界平移
                    rot_head2world=R_head2world,  # 云台到世界旋转
                    trans_pnp2head=T_pnp2head,  # PnP到云台平移
                    rot_pnp2head=R_pnp2head,  # PnP到云台旋转
                    use_plus_pnp=True  # 是否使用增强PnP
                )

                self.pnp_result_info = {
                    'pos': calc_pos,  # PnP计算的位置
                    'rpy': calc_rpy,  # PnP计算的rpy
                    'true_pos': armor.world_pos,  # 观测真值
                    'pixel_points': ordered_pixels,  # 用于绘制的像素点
                    'corner_points_3d': corners_3d_world,  # 3D角点位置
                    'armor_type': 'large',
                }

                obsrv_armor = copy.deepcopy(armor)

                pos_noise = np.random.normal(0, self.pos_noise_sigma, 3)
                rpy_noise = np.random.normal(0, self.rpy_noise_sigma, 3)
                yaw_noise = np.random.normal(0, self.rpy_noise_sigma, 1)

                obsrv_armor.world_pos = calc_pos + pos_noise
                # print(obsrv_armor.world_rpy, calc_rpy)
                # print(limit_rad(obsrv_armor.world_rpy - calc_rpy))
                obsrv_armor.world_rpy = calc_rpy + rpy_noise

                obsrv_armors.append(obsrv_armor)

        self.obsrv_data_with_t = (
            obsrv_armors,
            time.time()
        )

    def get_intrinsic_matrix(self):
        return np.array([
            [self.camera.focal_len, 0, 0],
            [0, self.camera.focal_len, 0],
            [0, 0, 1]
        ])




