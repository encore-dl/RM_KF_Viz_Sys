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

# 世界坐标系：x 右 y 前 z 上
# 相机坐标系：x 右 y 下 z 前
# 图像坐标系：x 右 y 下
# 像素坐标系：u 右 v 下
# 空间旋转为 x roll y pitch z yaw -> xyz rpy
class Visualizer:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pg.display.set_mode((self.screen_width, self.screen_height))

        self.main_screen_width = self.screen_width // 3 * 2
        self.main_screen_height = self.screen_height
        self.camera_screen_width = self.screen_width // 3
        self.camera_screen_height = self.screen_height // 2

        self.main_screen_center = np.array([self.screen_width // 3, self.screen_height // 2])
        self.camera_screen_center = np.array([self.screen_width // 6 * 5, self.screen_height // 6])

        self.world_scale = WORLD_SCALE

    def show(self, true_robots, obsrv_armors, tracker):
        self.screen.fill(Color.BLACK)

        self.show_main(true_robots, obsrv_armors, tracker)
        self.show_camera()
        self.show_info()

    def show_main(self, true_robots, obsrv_armors, tracker):
        # 画真实装甲板
        # 也就是真实数据 true data
        for robot in true_robots:
            # 车，装甲板的可视化
            robot_main_screen_pos = world_to_main_screen(
                world_pos=robot.world_pos,
                screen_center=self.main_screen_center,
                world_scale=self.world_scale
            )
            # print(robot_main_screen_pos)
            pg.draw.circle(self.screen, Color.BLUE, robot_main_screen_pos, 6)

            for armor in robot.armors:
                armor_main_screen_pos = world_to_main_screen(
                    armor.world_pos,
                    self.main_screen_center,
                    self.world_scale
                )
                print(armor_main_screen_pos)
                pg.draw.circle(self.screen, Color.GREEN, armor_main_screen_pos, 4)

        # 画 加了高斯噪声的装甲板
        # 也就是观测数据 obsrv
        for obsrv_armor in obsrv_armors:
            obsrv_armor_main_screen_pos = world_to_main_screen(
                obsrv_armor.world_pos,
                self.main_screen_center,
                self.world_scale
            )
            pg.draw.circle(self.screen, Color.YELLOW, obsrv_armor_main_screen_pos, 5)

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

    def show_camera(self):
        pass

    def show_info(self):
        pass







