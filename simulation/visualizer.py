import numpy as np
import pygame as pg
import math
import random

from dataclasses import dataclass

from utils.math_tools import world_to_main_screen


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


SCREEN_WIDTH = 1500
SCREEN_HEIGHT = 840

WORLD_SCALE = 10
OBSRV_NOISE = 0.1

PI = math.pi


class Visualizer:
    def __init__(self):
        self.screen_width = SCREEN_WIDTH
        self.screen_height = SCREEN_HEIGHT
        self.screen = pg.display.set_mode((self.screen_width, self.screen_height))

        self.main_screen_center = np.array([self.screen_width // 3, self.screen_height // 2])
        self.camera_screen_center = np.array([self.screen_width // 6 * 5, self.screen_height // 6 * 5])

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
                armor_main_pos = world_to_main_screen(
                    [
                        robot.world_pos[0] - armor.radius * math.cos(armor_main_angle),
                        robot.world_pos[1] - armor.radius * math.sin(armor_main_angle)
                    ],
                    self.main_screen_center,
                    self.world_scale
                )

                pg.draw.circle(self.screen, Color.GREEN, armor_main_pos, 4)

                # 画加了高斯噪声的装甲板
                # 也就是观测数据 observe
                armor_main_obsrv_pos = armor_main_pos + np.array([random.gauss(0, OBSRV_NOISE), random.gauss(0, OBSRV_NOISE)])
                pg.draw.circle(self.screen, Color.YELLOW, armor_main_obsrv_pos, 5)

            # if targets


    def show_camera(self):
        pass

    def show_info(self):
        pass







