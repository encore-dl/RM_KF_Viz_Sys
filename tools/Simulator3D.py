import pygame
import numpy as np
import time
import math

from RM_KF_Viz_Sys.track.Tracker import Tracker
from RM_KF_Viz_Sys.object.Motion import Motion
# from RM_KF_Viz_Sys.object.Camera import Camera

from RM_KF_Viz_Sys.tools.math_tools import world_to_screen
from RM_KF_Viz_Sys.tools.GenerData import generate_armors, update_true_target
from RM_KF_Viz_Sys.tools.draw_rel.draw_tools import draw_scale, draw_legend, draw_state


class Simulator3D:
    """3D模拟器主类 - 灵活的运动函数接口"""

    def __init__(self):
        # 创建真实目标
        self.true_target = {
            'center': np.array([0.0, 0.0, 1.0]),  # 旋转中心 [x, y, z] (米)
            'vel': np.array([0.0, 0.0, 0.0]),  # 速度 [vx, vy, vz] (米/秒)
            'angle': 0.0,  # 角度 (弧度)
            'angular_vel': 0.0,  # 角速度 (弧度/秒)
            'radius': 0.2,  # 半径 (米)
            'armor_num': 4,
            'armor_height': 0.1  # 装甲板高度差
        }
        self.auto_track_target = False

        # self.camera = Camera(
        #     position=(-1.7, -1.3, 1),  # 相机在原点
        #     fov=60,  # 60度视野
        #     max_range=8,  # 8米检测距离
        #     orientation=(PI / 4, 0, 0),
        # )

        self.tracker = Tracker()
        self.observation_noise_pos = 0.01  # 位置观测噪声 (米)
        self.observation_noise_angle = 0.05  # 角度观测噪声 (弧度)
        self.detection_prob = 1.0  # 检测概率
        self.last_time = time.time()

        # 运动模式配置 - 每个模式包含持续时间和运动函数
        self.current_pattern = 0
        self.pattern_start_time = time.time()

        # ========== 在这里添加你的运动函数 ==========
        self.motion = Motion()
        self.motion_patterns = [
            {"duration": 6.0, "motion_func": self.motion.horizontal_oscillate},
            {"duration": 6.0, "motion_func": self.motion.vertical_oscillate},
            {"duration": 6.0, "motion_func": self.motion.diagonal_oscillate},
            {"duration": 3.0, "motion_func": self.motion.stationary},
            {"duration": 6.0, "motion_func": self.motion.horizontal_oscillate_rotate},
            {"duration": 6.0, "motion_func": self.motion.vertical_oscillate_rotate},
            {"duration": 6.0, "motion_func": self.motion.diagonal_oscillate_rotate},
            {"duration": 3.0, "motion_func": self.motion.rotate_in_place}
        ]
        # ==========================================

        # 坐标转换参数
        self.SCREEN_WIDTH = 1200
        self.SCREEN_HEIGHT = 800
        self.world_scale = 100  # 1米 = 200像素
        self.screen_center = np.array([self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2])

        self.color = {
            'WHITE': (255, 255, 255),
            'BLACK': (0, 0, 0),
            'RED': (255, 0, 0),
            'GREEN': (0, 255, 0),
            'BLUE': (0, 0, 255),
            'YELLOW': (255, 255, 0),
            'PURPLE': (128, 0, 128),
            'CYAN': (0, 255, 255),
        }
        self.PI = math.pi

    def update(self):
        """更新模拟器"""
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        # self.camera.update(dt)
        # if self.auto_track_target:
        #     self.camera.look_at(self.true_target['center'])

        # 更新真实目标
        update_true_target(self, dt)

        # 生成观测
        armors = generate_armors(self)

        # 跟踪
        targets = self.tracker.track(armors, current_time)

        return armors, targets

    def draw(self, screen, armors, targets):
        """绘制模拟器状态 - 只在XY平面可视化"""
        # 清屏
        screen.fill(self.color['BLACK'])

        self.draw_camera_view(screen, armors)

        self.draw_main_view(screen, armors, targets)

        self.draw_info(screen, targets)

    def draw_info(self, screen, targets):
        font = pygame.font.SysFont(None, 24)
        
        draw_state(self, screen, font, targets)

        # === 图例 ===
        draw_legend(self, screen, font)

        # === 比例尺 ===
        draw_scale(self, screen, font)

    def draw_camera_view(self, screen, armors):
        """绘制相机视图（如果Camera类已实现）"""
        # 这里可以添加相机视图的绘制逻辑
        pass