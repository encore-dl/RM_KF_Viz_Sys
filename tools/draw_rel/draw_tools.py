import pygame
import time
import math

from RM_KF_Viz_Sys.tools.math_tools import world_to_screen


def draw_state(self, screen, font, targets):
    """绘制状态信息面板"""

    # === 基础状态信息 ===
    state_text = f"State: {self.tracker.state_}"
    state_surface = font.render(state_text, True, self.color['WHITE'])
    screen.blit(state_surface, (10, 10))

    # === NIS 信息 ===
    if self.tracker.target_:
        nis_value = self.tracker.target_.ekf().data.get('nis', 0)
        nis_failures = self.tracker.target_.ekf().data.get('recent_nis_failures', 0)
        window_size = self.tracker.target_.ekf().window_size

        nis_text = f"NIS: {nis_value:.2f}"
        nis_fail_text = f"NIS Failures: {nis_failures}/{window_size}"
    else:
        nis_text = "NIS: -"
        nis_fail_text = "NIS Failures: -"

    nis_surface = font.render(nis_text, True, self.color['WHITE'])
    screen.blit(nis_surface, (10, 40))

    nis_fail_surface = font.render(nis_fail_text, True, self.color['WHITE'])
    screen.blit(nis_fail_surface, (10, 70))

    # === 运动模式信息 ===
    current_pattern = self.motion_patterns[self.current_pattern]
    pattern_elapsed = time.time() - self.pattern_start_time
    pattern_text = (f"Motion: {self.current_pattern}/7 | "
                    f"Time: {pattern_elapsed:.1f}/{current_pattern['duration']}s")
    pattern_surface = font.render(pattern_text, True, self.color['WHITE'])
    screen.blit(pattern_surface, (10, 100))

    # === 三维坐标信息 ===
    true_z = self.true_target['center'][2]
    if targets and self.tracker.target_:
        est_z = self.tracker.target_.ekf_x()[4]
        z_text = f"Z: {true_z:.2f}m (True), {est_z:.2f}m (Est)"
    else:
        z_text = f"Z: {true_z:.2f}m (True), - (Est)"

    z_surface = font.render(z_text, True, self.color['WHITE'])
    screen.blit(z_surface, (10, 130))

    # === 速度信息 ===
    true_vx, true_vy, true_vz = self.true_target['vel']
    if targets and self.tracker.target_:
        est_vx = self.tracker.target_.ekf_x()[1]
        est_vy = self.tracker.target_.ekf_x()[3]
        est_vz = self.tracker.target_.ekf_x()[5]

        v_text = (f"V: ({true_vx:.1f},{true_vy:.1f},{true_vz:.1f}) | "
                  f"Est: ({est_vx:.1f},{est_vy:.1f},{est_vz:.1f})")
    else:
        v_text = f"V: ({true_vx:.1f},{true_vy:.1f},{true_vz:.1f}) | Est: (-,-,-)"

    v_surface = font.render(v_text, True, self.color['WHITE'])
    screen.blit(v_surface, (10, 160))

    # === 角度信息 ===
    true_angle = self.true_target['angle']
    true_w = self.true_target['angular_vel']
    if targets and self.tracker.target_:
        est_angle = self.tracker.target_.ekf_x()[6]
        est_w = self.tracker.target_.ekf_x()[7]
        angle_text = (f"Angle: {math.degrees(true_angle):.1f}° | W: {true_w:.2f} | "
                      f"Est: {math.degrees(est_angle):.1f}° | {est_w:.2f}")
    else:
        angle_text = f"Angle: {math.degrees(true_angle):.1f}° | W: {true_w:.2f} | Est: -"

    angle_surface = font.render(angle_text, True, self.color['WHITE'])
    screen.blit(angle_surface, (10, 190))


def draw_legend(self, screen, font):
    """绘制图例"""
    legend_y = self.SCREEN_HEIGHT - 180
    legend_items = [
        ("True Center", self.color['BLUE']),
        ("True Armor (Green=Low, Bright=High)", self.color['GREEN']),
        ("Observed Armor", self.color['YELLOW']),
        ("Estimated Center", self.color['RED']),
        ("Estimated Armor", self.color['PURPLE'])
    ]

    for i, (text, color) in enumerate(legend_items):
        y_pos = legend_y + i * 25

        if "Armor" in text and "True" in text:
            # 对于真实装甲板，绘制两个不同颜色的点表示高度变化
            pygame.draw.circle(screen, (0, 100, 0), (20, y_pos), 4)
            pygame.draw.circle(screen, (0, 200, 0), (30, y_pos), 4)
        else:
            pygame.draw.circle(screen, color, (20, y_pos), 5)

        text_surface = font.render(text, True, self.color['WHITE'])
        screen.blit(text_surface, (45, y_pos - 8))


def draw_scale(self, screen, font):
    """绘制比例尺"""
    scale_x = self.SCREEN_WIDTH - 120
    scale_y = self.SCREEN_HEIGHT - 60

    # 绘制1米比例尺
    scale_length = self.world_scale  # 1米对应的像素长度
    pygame.draw.line(screen, self.color['WHITE'],
                     (scale_x, scale_y),
                     (scale_x + scale_length, scale_y), 2)

    # 绘制刻度标记
    pygame.draw.line(screen, self.color['WHITE'],
                     (scale_x, scale_y - 5),
                     (scale_x, scale_y + 5), 2)
    pygame.draw.line(screen, self.color['WHITE'],
                     (scale_x + scale_length, scale_y - 5),
                     (scale_x + scale_length, scale_y + 5), 2)

    # 添加文字
    scale_text = "1 meter"
    scale_surface = font.render(scale_text, True, self.color['WHITE'])
    screen.blit(scale_surface, (scale_x + scale_length // 2 - 25, scale_y + 10))


def draw_main_view(self, screen, armors, targets):
    # 绘制真实目标中心
    true_center_xy = self.true_target['center'][:2]  # 只取XY
    true_center_screen = world_to_screen(self, true_center_xy)
    pygame.draw.circle(screen, self.color['BLUE'], true_center_screen, 6)

    # 绘制真实装甲板
    center = self.true_target['center']
    angle = self.true_target['angle']
    radius = self.true_target['radius']
    armor_num = self.true_target['armor_num']

    for i in range(armor_num):
        armor_angle = angle + i * 2 * PI / armor_num
        armor_x = center[0] - radius * math.cos(armor_angle)
        armor_y = center[1] - radius * math.sin(armor_angle)
        armor_screen = world_to_screen(self, [armor_x, armor_y])

        # 根据Z坐标调整颜色深浅表示高度
        z_ratio = center[2] / 2.0  # 假设最大高度2米
        color_intensity = int(100 + 155 * z_ratio)
        armor_color = (0, color_intensity, 0)  # 绿色，根据高度变化

        pygame.draw.circle(screen, armor_color, armor_screen, 4)

    # 绘制观测到的装甲板
    for armor in armors:
        armor_screen = (armor.center[0], armor.center[1])
        pygame.draw.circle(screen, self.color['YELLOW'], armor_screen, 5)

    # 绘制跟踪结果
    if targets and self.tracker.state_ != "lost":
        target = targets[0]

        # 估计的旋转中心
        est_center_xy = [target.ekf_x()[0], target.ekf_x()[2]]  # x, y
        est_center_screen = world_to_screen(self, est_center_xy)
        pygame.draw.circle(screen, self.color['RED'], est_center_screen, 6)

        # 估计的装甲板位置
        for i in range(target.armor_num_):
            armor_xyz = target.h_armor_xyz(target.ekf_x(), i)
            armor_screen = world_to_screen(self, armor_xyz[:2])  # 只取XY
            pygame.draw.circle(screen, self.color['PURPLE'], armor_screen, 4)