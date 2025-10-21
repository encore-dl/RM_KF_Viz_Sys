import numpy as np
import pygame as pg
import math
import random

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
OBSRV_NOISE = 0.1

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

    def show(self, robots):
        self.screen.fill(Color.BLACK)

        self.show_main(robots)
        self.show_camera()
        self.show_info()

    def show_main(self, robots, targets=None):
        for robot in robots:
            # 车，装甲板的可视化
            world_xy = robot.world_pos[:2]
            main_xy = world_to_main_screen(world_pos=world_xy, screen_center=self.main_screen_center, world_scale=self.world_scale)
            pg.draw.circle(self.screen, Color.BLUE, main_xy, 6)

            for armor in robot.armors:
                # 画真实的装甲板
                # 也就是真实数据 true
                armor_main_angle = robot.world_rpy[0] + armor.armor_id * 2*PI / robot.armor_count
                armor_main_screen_pos = world_to_main_screen(
                    [
                        robot.world_pos[0] - armor.radius * math.cos(armor_main_angle),
                        robot.world_pos[1] - armor.radius * math.sin(armor_main_angle)  # y轴本就是向下的
                    ],
                    self.main_screen_center,
                    self.world_scale
                )

                pg.draw.circle(self.screen, Color.GREEN, armor_main_screen_pos, 4)

                # 画加了高斯噪声的装甲板
                # 也就是观测数据 observe
                armor_main_obsrv_pos = armor_main_screen_pos + np.array([random.gauss(0, OBSRV_NOISE), random.gauss(0, OBSRV_NOISE)])
                pg.draw.circle(self.screen, Color.YELLOW, armor_main_obsrv_pos, 5)

            # if targets

    def show_camera(self):
        pass

    def show_info(self):
        pass







