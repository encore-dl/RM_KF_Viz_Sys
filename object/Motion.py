import math
import numpy as np

class Motion:
    def stationary(self, t):
        """原地静止"""
        return {
            'pos': np.array([0.0, 0.0, 1.0]),
            'vel': np.array([0.0, 0.0, 0.0]),
            'angular_vel': 0.0
        }


    def horizontal_oscillate(self, t):
        """左右横跳回到中心 - 符合惯性"""
        ampl = 2.0
        freq = 0.25

        # 平滑的横跳运动，在端点减速，在中心加速
        x = (ampl * (1 - math.cos(4 * math.pi * freq * t)) ** 2 / 4) if 0 <= (t % 4 * math.pi) < 2 * math.pi else -(
                    ampl * (1 - math.cos(4 * math.pi * freq * t)) ** 2 / 4)
        vx = (ampl * math.pi * freq * math.sin(2 * math.pi * freq * t) * (1 - math.cos(2 * math.pi * freq * t))) if 0 <= (
                    t % 4 * math.pi) < 2 * math.pi else -(
                    ampl * math.pi * freq * math.sin(2 * math.pi * freq * t) * (1 - math.cos(2 * math.pi * freq * t)))

        return {
            'pos': np.array([x, 0.0, 1.0]),
            'vel': np.array([vx, 0.0, 0.0]),
            'angular_vel': 0.0
        }


    def vertical_oscillate(self, t):
        """上下横跳回到中心 - 符合惯性"""
        ampl = 2.0
        freq = 0.25

        y = (ampl * (1 - math.cos(4 * math.pi * freq * t)) ** 2 / 4) if 0 <= (t % 4 * math.pi) < 2 * math.pi else -(
                    ampl * (1 - math.cos(4 * math.pi * freq * t)) ** 2 / 4)
        vy = (ampl * math.pi * freq * math.sin(2 * math.pi * freq * t) * (1 - math.cos(2 * math.pi * freq * t))) if 0 <= (
                    t % 4 * math.pi) < 2 * math.pi else -(
                    ampl * math.pi * freq * math.sin(2 * math.pi * freq * t) * (1 - math.cos(2 * math.pi * freq * t)))

        return {
            'pos': np.array([0.0, y, 1.0]),
            'vel': np.array([0.0, vy, 0.0]),
            'angular_vel': 0.0
        }


    def diagonal_oscillate(self, t):
        """斜向横跳回到中心 - 符合惯性"""
        ampl = 2.0
        freq = 0.25

        # 45度斜向运动
        factor = 0.707  # 1/sqrt(2)
        x = (ampl * (1 - math.cos(4 * math.pi * freq * t)) ** 2 / 4) if 0 <= (t % 4 * math.pi) < 2 * math.pi else -(
                    ampl * (1 - math.cos(4 * math.pi * freq * t)) ** 2 / 4)
        y = (ampl * (1 - math.cos(4 * math.pi * freq * t)) ** 2 / 4) if 0 <= (t % 4 * math.pi) < 2 * math.pi else -(
                    ampl * (1 - math.cos(4 * math.pi * freq * t)) ** 2 / 4)

        vx = (ampl * math.pi * freq * math.sin(2 * math.pi * freq * t) * (1 - math.cos(2 * math.pi * freq * t))) if 0 <= (
                    t % 4 * math.pi) < 2 * math.pi else -(
                    ampl * math.pi * freq * math.sin(2 * math.pi * freq * t) * (1 - math.cos(2 * math.pi * freq * t)))
        vy = (ampl * math.pi * freq * math.sin(2 * math.pi * freq * t) * (1 - math.cos(2 * math.pi * freq * t))) if 0 <= (
                    t % 4 * math.pi) < 2 * math.pi else -(
                    ampl * math.pi * freq * math.sin(2 * math.pi * freq * t) * (1 - math.cos(2 * math.pi * freq * t)))

        return {
            'pos': np.array([x, y, 1.0]),
            'vel': np.array([vx, vy, 0.0]),
            'angular_vel': 0.0
        }


    def horizontal_oscillate_rotate(self, t):
        """左右横跳+旋转"""
        base_motion = self.horizontal_oscillate(t)
        base_motion['angular_vel'] = 2.7
        return base_motion


    def vertical_oscillate_rotate(self, t):
        """上下横跳+旋转"""
        base_motion = self.vertical_oscillate(t)
        base_motion['angular_vel'] = 2.7
        return base_motion


    def diagonal_oscillate_rotate(self, t):
        """斜向横跳+旋转"""
        base_motion = self.diagonal_oscillate(t)
        base_motion['angular_vel'] = 2.7
        return base_motion


    def rotate_in_place(self, t):
        """原地旋转"""
        return {
            'pos': np.array([0.0, 0.0, 1.0]),
            'vel': np.array([0.0, 0.0, 0.0]),
            'angular_vel': 2.7
        }