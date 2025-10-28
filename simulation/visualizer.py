import numpy as np
import pygame as pg
import math

from dataclasses import dataclass

from utils.math_tool import world_to_main_screen


@dataclass
class Color:
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    PURPLE = (128, 0, 128)
    CYAN = (0, 255, 255)


WORLD_SCALE = 2

PI = math.pi


# 基本定义
# pygame的screen坐标系是：x 右 y 下

# 世界坐标系：x 前 y 右 z 上
# 相机坐标系：x 右 y 下 z 前
# 图像坐标系：x 右 y 下
# 像素坐标系：u 右 v 下
# 空间旋转为 前 roll 右 pitch 上 yaw
class Visualizer:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pg.display.set_mode((self.screen_width, self.screen_height))

        self.main_screen_width = self.screen_width // 3 * 2
        self.main_screen_height = self.screen_height
        self.camera_screen_width = self.screen_width // 3
        self.camera_screen_height = self.screen_height // 2
        self.info_screen_width = self.screen_width // 3
        self.info_screen_height = self.screen_height // 2

        self.main_screen_center = np.array([self.screen_width // 3, self.screen_height // 2])
        self.camera_screen_center = np.array([self.screen_width // 6 * 5, self.screen_height // 6])

        self.world_scale = WORLD_SCALE

    def show(self, true_robots, obsrv_armors, tracker, camera):
        self.screen.fill(Color.BLACK)

        self.show_main(true_robots, obsrv_armors, tracker, camera)
        self.show_camera()
        self.show_info()

    def show_main(self, true_robots, obsrv_armors, tracker, camera):
        # 画真实装甲板
        # 也就是真实数据 true data
        for robot in true_robots:
            # 车，装甲板的可视化
            robot_main_screen_pos = world_to_main_screen(
                world_pos=robot.world_pos,
                screen_center=self.main_screen_center,
                world_scale=self.world_scale
            )
            pg.draw.circle(self.screen, Color.BLUE, robot_main_screen_pos, 6)

            for armor in robot.armors:
                armor_main_screen_pos = world_to_main_screen(
                    armor.world_pos,
                    self.main_screen_center,
                    self.world_scale
                )
                pg.draw.circle(self.screen, Color.WHITE, armor_main_screen_pos, 4)

        # 画 加了高斯噪声的装甲板
        # 也就是观测数据 obsrv
        for obsrv_armor in obsrv_armors:
            obsrv_armor_main_screen_pos = world_to_main_screen(
                obsrv_armor.world_pos,
                self.main_screen_center,
                self.world_scale
            )
            pg.draw.circle(self.screen, Color.YELLOW, obsrv_armor_main_screen_pos, 5)

        # 画 模型导出的数据
        # 也就是 估计数据 est
        if tracker.is_tracked:
            model = tracker.tongji_model

            est_center_main_screen_pos = world_to_main_screen(
                [
                    model.get_ekf().x[0],
                    model.get_ekf().x[2]
                ],
                self.main_screen_center,
                self.world_scale
            )
            pg.draw.circle(self.screen, Color.RED, est_center_main_screen_pos, 6)
            
            for armor_id in range(tracker.tracked_robot.armor_count):
                est_armor_pos = model.get_est_armor_pos(model.get_ekf().x, armor_id)
                est_armor_main_screen_pos = world_to_main_screen(
                    est_armor_pos,
                    self.main_screen_center,
                    self.world_scale
                )
                pg.draw.circle(self.screen, Color.PURPLE, est_armor_main_screen_pos, 4)

        # 绘制相机在 main screen 上的位置
        camera_main_screen_pos = world_to_main_screen(
            camera.world_pos,
            self.main_screen_center,
            self.world_scale
        )
        pg.draw.circle(self.screen, Color.CYAN, camera_main_screen_pos, 8)

        # 绘制扇形视线
        forward_vec = camera.get_forward_vec()
        forward_end = camera.world_pos + forward_vec * 30
        forward_main_screen_pos = world_to_main_screen(
            forward_end,
            self.main_screen_center,
            self.world_scale
        )
        pg.draw.line(self.screen, Color.CYAN, camera_main_screen_pos, forward_main_screen_pos, 3)

        # 绘制视野的扇形区域
        fov = camera.fov
        max_range = camera.max_range
        forward_yaw = math.atan2(forward_vec[1], forward_vec[0])

        fan_seg_count = 20  # 扇形绘制的平滑程度
        fan_vertexes = [camera_main_screen_pos]

        # 寻找扇缘的平滑度分隔点
        for i in range(fan_seg_count + 1):
            fan_seg_agl = forward_yaw - fov/2 + ((fov / fan_seg_count) * i)

            fan_vertex_world_pos = camera.world_pos.copy()
            fan_vertex_world_pos[0] += math.cos(fan_seg_agl) * max_range
            fan_vertex_world_pos[1] += math.sin(fan_seg_agl) * max_range

            fan_vertex_main_screen_pos = world_to_main_screen(
                fan_vertex_world_pos,
                self.main_screen_center,
                self.world_scale
            )
            fan_vertexes.append(fan_vertex_main_screen_pos)

        if len(fan_vertexes) >= 3:
            # 绘制扇形
            transparent_surface = pg.Surface(
                (self.main_screen_width, self.main_screen_height),
                pg.SRCALPHA
            )
            pg.draw.polygon(transparent_surface, (0, 255, 255, 30), fan_vertexes)
            self.screen.blit(transparent_surface, (0, 0))
            pg.draw.lines(self.screen, Color.CYAN, False, fan_vertexes[1:], 2)

            # 绘制扇形对称线
            fan_mid_world_pos = camera.world_pos.copy()
            fan_mid_world_pos[0] += math.cos(forward_yaw) * max_range
            fan_mid_world_pos[1] += math.sin(forward_yaw) * max_range
            fan_mid_main_screen_pos = world_to_main_screen(
                fan_mid_world_pos,
                self.main_screen_center,
                self.world_scale
            )
            pg.draw.line(self.screen, Color.CYAN, camera_main_screen_pos, fan_mid_main_screen_pos, 1)

    def show_camera(self):
        pass

    def show_info(self):
        pass







