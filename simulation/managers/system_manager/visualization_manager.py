import threading
import numpy as np
import pygame as pg
import math
from dataclasses import dataclass
from collections import deque

from simulation.event_bus import event_bus
from simulation.dataflow import Observation, PnPResult, Prediction, DrawText
from core.algorithms.math import world_to_main_screen, world_to_camera_screen


@dataclass
class Color:
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    PURPLE = (255, 0, 255)
    VIOLET = (100, 100, 255)
    CYAN = (0, 255, 255)
    ORANGE = (255, 165, 0)


WORLD_SCALE = 200
PI = math.pi


class VisualizationManager:
    def __init__(self, screen_width, screen_height, robot_manager):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pg.display.set_mode((self.screen_width, self.screen_height))
        self.world_scale = WORLD_SCALE
        self.robot_manager = robot_manager

        self.main_screen_width = self.screen_width // 3 * 2
        self.main_screen_height = self.screen_height
        self.camera_screen_width = self.screen_width // 3
        self.camera_screen_height = self.screen_height // 3
        self.info_screen_width = self.screen_width // 3
        self.info_screen_height = self.screen_height // 3 * 2

        self.main_screen_center = np.array([self.screen_width // 3, self.screen_height // 2])
        self.camera_screen_center = np.array([self.screen_width // 6 * 5, self.screen_height // 6])
        self.info_screen_center = np.array([self.screen_width // 6 * 5, self.screen_height // 3 * 2])

        self._latest_obs = None
        self._latest_pnp_list = []
        self._latest_pred = None
        self._draw_texts = deque(maxlen=10)
        self._lock = threading.Lock()

        event_bus.subscribe('obs', self._on_obs)
        event_bus.subscribe('pnp', self._on_pnp)
        event_bus.subscribe('pred', self._on_pred)
        event_bus.subscribe('draw', self._on_draw)

    def _on_obs(self, data: Observation):
        with self._lock:
            self._latest_obs = data

    def _on_pnp(self, data: PnPResult):
        with self._lock:
            self._latest_pnp_list = data

    def _on_pred(self, data: Prediction):
        with self._lock:
            self._latest_pred = data

    def _on_draw(self, data: DrawText):
        with self._lock:
            self._draw_texts.append(data)

    def show(self, bullets):
        self.screen.fill(Color.BLACK)

        with self._lock:
            obs = self._latest_obs
            pnp_list = self._latest_pnp_list
            pred = self._latest_pred

        robots = self.robot_manager.robots
        selected_robot = self.robot_manager.selected_robot
        selected_camera = selected_robot.get_camera() if selected_robot else None

        self.show_main_screen(robots, selected_camera, obs, pred, pnp_list, bullets)
        self.show_camera_screen(robots, selected_camera, obs, pred, pnp_list, bullets)
        self.show_info_screen(robots, obs, pred, pnp_list)

    def show_main_screen(self, robots, selected_camera, obs, pred, pnp_list, bullets):
        main_screen_rect = pg.Rect(
            self.main_screen_center[0] - self.main_screen_width // 2,
            self.main_screen_center[1] - self.main_screen_height // 2,
            self.main_screen_width,
            self.main_screen_height
        )
        pg.draw.rect(self.screen, (20, 20, 20), main_screen_rect)
        pg.draw.rect(self.screen, Color.WHITE, main_screen_rect, 2)

        # 绘制所有相机位置（即云台位置）
        for robot in robots:
            cam = robot.get_camera()
            color = Color.CYAN if cam == selected_camera else Color.VIOLET
            self._draw_point_main(cam.world_pos, color, 6)

        # 画真实装甲板
        for robot in robots:
            self._draw_point_main(robot.chassis.world_pos, Color.BLUE, 6)
            for armor in robot.get_armors():
                self._draw_point_main(armor.world_pos, Color.WHITE, 4)
                self._draw_armor_box_main(armor, Color.CYAN)
                for ep in armor.light_corners:
                    self._draw_point_main(ep, Color.CYAN, 4)

        # 观测数据
        if obs:
            for obs_armor in obs.obs_armors:
                self._draw_point_main(obs_armor.world_pos, Color.YELLOW, 5)

        # 预测数据
        if pred and pred.is_tracking:
            if pred.center is not None:
                self._draw_point_main(pred.center, Color.RED, 6)
            for pred_armor_pos in pred.armors:
                self._draw_point_main(pred_armor_pos, Color.PURPLE, 4)

        # PnP结果
        for pnp in pnp_list:
            if pnp and pnp.pnp_pos is not None and not np.isnan(pnp.pnp_pos).any():
                center = self._draw_point_main(pnp.pnp_pos, Color.ORANGE, 0)
                if center is not None:
                    pg.draw.line(self.screen, Color.ORANGE, (center[0] - 5, center[1] - 5), (center[0] + 5, center[1] + 5), 2)
                    pg.draw.line(self.screen, Color.ORANGE, (center[0] + 5, center[1] - 5), (center[0] - 5, center[1] + 5), 2)

        # 子弹
        if bullets:
            for bullet in bullets:
                if bullet.hit:
                    center = self._draw_point_main(bullet.hit_pos, Color.RED, 0)
                    if center is not None:
                        pg.draw.line(self.screen, Color.RED, (center[0] - 8, center[1] - 8), (center[0] + 8, center[1] + 8), 3)
                        pg.draw.line(self.screen, Color.RED, (center[0] + 8, center[1] - 8), (center[0] - 8, center[1] + 8), 3)
                elif bullet.active:
                    self._draw_point_main(bullet.pos, Color.GREEN, 3)

        if selected_camera:
            self._draw_camera_fov(selected_camera)

    def _draw_camera_fov(self, camera):
        camera_main_screen_pos = self._draw_point_main(camera.world_pos, Color.CYAN, 8)
        forward_vec = camera.get_forward_vec()
        forward_end = camera.world_pos + forward_vec * 30
        forward_main_screen_pos = world_to_main_screen(
            forward_end,
            self.main_screen_center,
            self.world_scale
        )
        pg.draw.line(self.screen, Color.CYAN, camera_main_screen_pos, forward_main_screen_pos, 3)

        fov = camera.fov
        max_range = camera.max_range
        forward_yaw = math.atan2(forward_vec[1], forward_vec[0])

        fan_seg_count = 20
        fan_vertexes = [camera_main_screen_pos]

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
            transparent_surface = pg.Surface(
                (self.main_screen_width, self.main_screen_height),
                pg.SRCALPHA
            )
            pg.draw.polygon(transparent_surface, (0, 255, 255, 30), fan_vertexes)
            self.screen.blit(transparent_surface, (0, 0))
            pg.draw.lines(self.screen, Color.CYAN, False, fan_vertexes[1:], 2)

            fan_mid_world_pos = camera.world_pos.copy()
            fan_mid_world_pos[0] += math.cos(forward_yaw) * max_range
            fan_mid_world_pos[1] += math.sin(forward_yaw) * max_range
            fan_mid_main_screen_pos = world_to_main_screen(
                fan_mid_world_pos,
                self.main_screen_center,
                self.world_scale
            )
            pg.draw.line(self.screen, Color.CYAN, camera_main_screen_pos, fan_mid_main_screen_pos, 1)

    def _draw_point_main(self, world_pos, color, radius):
        main_screen_pos = world_to_main_screen(
            world_pos=world_pos,
            main_screen_center=self.main_screen_center,
            world_scale=self.world_scale
        )
        pg.draw.circle(self.screen, color, main_screen_pos, radius)
        return main_screen_pos

    def _draw_armor_box_main(self, armor, color):
        main_screen_rect = pg.Rect(
            self.main_screen_center[0] - self.main_screen_width // 2,
            self.main_screen_center[1] - self.main_screen_height // 2,
            self.main_screen_width,
            self.main_screen_height
        )
        screen_points = []
        for corner in armor.light_corners:
            pos = world_to_main_screen(
                corner,
                self.main_screen_center,
                self.world_scale
            )
            if pos is None:
                return
            screen_points.append(pos)
        if len(screen_points) == 4:
            temp_surf = pg.Surface((main_screen_rect.width, main_screen_rect.height), pg.SRCALPHA)
            rel_points = [(p[0] - main_screen_rect.left, p[1] - main_screen_rect.top) for p in screen_points]
            pg.draw.polygon(temp_surf, (*color[:3], 80), rel_points)
            self.screen.blit(temp_surf, main_screen_rect)
            pg.draw.polygon(self.screen, color, screen_points, 2)

    def show_camera_screen(self, robots, selected_camera, obs, pred, pnp_list, bullets):
        camera_screen_rect = pg.Rect(
            self.camera_screen_center[0] - self.camera_screen_width // 2,
            self.camera_screen_center[1] - self.camera_screen_height // 2,
            self.camera_screen_width,
            self.camera_screen_height
        )
        pg.draw.rect(self.screen, (5, 5, 5), camera_screen_rect)
        pg.draw.rect(self.screen, Color.WHITE, camera_screen_rect, 2)

        if selected_camera is None:
            return

        for robot in robots:
            self._draw_point_camera(robot.chassis.world_pos, selected_camera, Color.BLUE, 6)
            for armor in robot.get_armors():
                self._draw_point_camera(armor.world_pos, selected_camera, Color.WHITE, 4)
                self._draw_armor_box_camera(armor, selected_camera, Color.CYAN)
                for ep in armor.light_corners:
                    self._draw_point_camera(ep, selected_camera, Color.CYAN, 4)

        if obs:
            for obs_armor in obs.obs_armors:
                self._draw_point_camera(obs_armor.world_pos, selected_camera, Color.YELLOW, 5)

        if pred and pred.is_tracking:
            if pred.center is not None:
                self._draw_point_camera(pred.center, selected_camera, Color.RED, 6)
            for pred_armor_pos in pred.armors:
                self._draw_point_camera(pred_armor_pos, selected_camera, Color.PURPLE, 4)

        for pnp in pnp_list:
            if pnp and pnp.pnp_pos is not None and not np.isnan(pnp.pnp_pos).any():
                center = self._draw_point_camera(pnp.pnp_pos, selected_camera, Color.ORANGE, 0)
                if center is not None:
                    pg.draw.line(self.screen, Color.ORANGE, (center[0] - 5, center[1] - 5), (center[0] + 5, center[1] + 5), 2)
                    pg.draw.line(self.screen, Color.ORANGE, (center[0] + 5, center[1] - 5), (center[0] - 5, center[1] + 5), 2)

        if bullets:
            for bullet in bullets:
                if bullet.hit:
                    center = self._draw_point_camera(bullet.hit_pos, selected_camera, Color.RED, 0)
                    if center is not None:
                        pg.draw.line(self.screen, Color.RED, (center[0] - 8, center[1] - 8), (center[0] + 8, center[1] + 8), 3)
                        pg.draw.line(self.screen, Color.RED, (center[0] + 8, center[1] - 8), (center[0] - 8, center[1] + 8), 3)
                elif bullet.active:
                    self._draw_point_camera(bullet.pos, selected_camera, Color.GREEN, 3)

    def _draw_point_camera(self, world_pos, camera, color, radius):
        resolution = (self.camera_screen_width, self.camera_screen_height)
        camera_screen_pos = world_to_camera_screen(
            world_pos=world_pos,
            camera=camera,
            camera_screen_center=self.camera_screen_center,
            resolution=resolution
        )
        if camera_screen_pos is not None:
            draw_pos = (int(camera_screen_pos[0]), int(camera_screen_pos[1]))
            pg.draw.circle(self.screen, color, draw_pos, radius)
        return camera_screen_pos

    def _draw_armor_box_camera(self, armor, camera, color):
        camera_screen_rect = pg.Rect(
            self.camera_screen_center[0] - self.camera_screen_width // 2,
            self.camera_screen_center[1] - self.camera_screen_height // 2,
            self.camera_screen_width,
            self.camera_screen_height
        )
        screen_points = []
        for corner in armor.light_corners:
            pos = world_to_camera_screen(
                corner, camera,
                self.camera_screen_center,
                (self.camera_screen_width, self.camera_screen_height)
            )
            if pos is None:
                return
            screen_points.append(pos)
        if len(screen_points) == 4:
            temp_surf = pg.Surface((camera_screen_rect.width, camera_screen_rect.height), pg.SRCALPHA)
            rel_points = [(p[0] - camera_screen_rect.left, p[1] - camera_screen_rect.top) for p in screen_points]
            pg.draw.polygon(temp_surf, (*color[:3], 80), rel_points)
            self.screen.blit(temp_surf, camera_screen_rect)
            pg.draw.polygon(self.screen, color, screen_points, 2)

    def show_info_screen(self, robots, obs, pred, pnp_list):
        info_screen_rect = pg.Rect(
            self.info_screen_center[0] - self.info_screen_width // 2,
            self.info_screen_center[1] - self.info_screen_height // 2,
            self.info_screen_width,
            self.info_screen_height
        )
        pg.draw.rect(self.screen, (50, 50, 50), info_screen_rect)
        pg.draw.rect(self.screen, Color.WHITE, info_screen_rect, 2)

        texts_colors = []

        def entry_append(text_, color_):
            texts_colors.append((text_, color_))

        if obs is not None:
            entry_append(f"obs armors: {len(obs.obs_armors)}", Color.CYAN)

        if pred is not None:
            entry_append(f"is tracking: {pred.is_tracking}", Color.GREEN)
            if pred.center is not None and len(pred.center) >= 3:
                entry_append(
                    f"pred center: ({pred.center[0]:.3f}, {pred.center[1]:.3f}, {pred.center[2]:.3f})",
                    Color.WHITE
                )
            else:
                entry_append("pred center: None", Color.WHITE)

            if pred.armors:
                for i, armor_pos in enumerate(pred.armors):
                    if armor_pos is not None and len(armor_pos) >= 3:
                        entry_append(
                            f"pred armor{i}: ({armor_pos[0]:.3f}, {armor_pos[1]:.3f}, {armor_pos[2]:.3f})",
                            Color.GREEN
                        )
            else:
                entry_append("pred armors: []", Color.GREEN)

            entry_append(f"fps: {pred.fps:.3f}", Color.GREEN)

            if pred.state_vector is not None:
                sv = pred.state_vector
                entry_append("state vector:", Color.YELLOW)
                display_len = min(len(sv), 11)
                indices = [0, 1, 2, 6, 7, 8, 9, 10]
                parts = []
                for idx in indices:
                    if idx < display_len:
                        parts.append(f"{sv[idx]:.3f}")
                if parts:
                    entry_append(f"  x,y,z,ψ,ω,ra,rb,dz: {', '.join(parts)}", Color.YELLOW)
                else:
                    entry_append(f"  {sv}", Color.YELLOW)

        with self._lock:
            draw_texts = list(self._draw_texts)
        if draw_texts:
            entry_append("--- Debug ---", Color.GREEN)
            for draw_text in draw_texts:
                entry_append(draw_text.text, draw_text.color)

        if not texts_colors:
            entry_append("No data", Color.WHITE)

        entry_count = len(texts_colors)
        font_size = 24
        max_entry_count = (info_screen_rect.height - 40) // (font_size + 8)
        while entry_count > max_entry_count and font_size > 12:
            font_size -= 2
            max_entry_count = (info_screen_rect.height - 40) // (font_size + 8)

        font = pg.font.SysFont(None, font_size)
        line_height = font_size + 8
        entry_per_col_count = max_entry_count
        col_count = max(1, math.ceil(entry_count / entry_per_col_count))
        col_width = info_screen_rect.width // col_count

        for col in range(col_count):
            start_idx = col * entry_per_col_count
            end_idx = min((col+1) * entry_per_col_count, entry_count)
            curr_y = info_screen_rect.top + 20
            for i, (text, color) in enumerate(texts_colors[start_idx:end_idx]):
                x = info_screen_rect.left + 10 + col * col_width
                text_width = font.size(text)[0]
                if text_width > col_width - 20:
                    words = text.split(' ')
                    broken_lines = []
                    curr_line = ""
                    for word in words:
                        test_line = curr_line + word + " "
                        if font.size(test_line)[0] <= col_width - 20:
                            curr_line = test_line
                        else:
                            if curr_line:
                                broken_lines.append(curr_line.strip())
                            curr_line = word + " "
                    if curr_line:
                        broken_lines.append(curr_line.strip())
                    for j, broken_line in enumerate(broken_lines):
                        surface = font.render(broken_line, True, color)
                        self.screen.blit(surface, (x, curr_y))
                        curr_y += line_height
                else:
                    surface = font.render(text, True, color)
                    self.screen.blit(surface, (x, curr_y))
                    curr_y += line_height