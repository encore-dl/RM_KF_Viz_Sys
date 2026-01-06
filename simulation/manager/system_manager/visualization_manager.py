import numpy as np
import pygame as pg
import math

from dataclasses import dataclass

from utils.math_tool import world_to_main_screen, world_to_camera_screen


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


WORLD_SCALE = 100

PI = math.pi


# 基本定义
# pygame的screen坐标系是：x 右 y 下

# 世界坐标系：x 前 y 右 z 上
# 相机坐标系：x 右 y 下 z 前
# 图像坐标系：x 右 y 下
# 像素坐标系：u 右 v 下
# 空间旋转为 前 roll 右 pitch 上 yaw
class VisualizationManager:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pg.display.set_mode((self.screen_width, self.screen_height))

        self.main_screen_width = self.screen_width // 3 * 2
        self.main_screen_height = self.screen_height
        self.camera_screen_width = self.screen_width // 3
        self.camera_screen_height = self.screen_height // 3
        self.info_screen_width = self.screen_width // 3
        self.info_screen_height = self.screen_height // 3 * 2

        self.main_screen_center = np.array([self.screen_width // 3, self.screen_height // 2])
        self.camera_screen_center = np.array([self.screen_width // 6 * 5, self.screen_height // 6])
        self.info_screen_center = np.array([self.screen_width // 6 * 5, self.screen_height // 3 * 2])

        self.world_scale = WORLD_SCALE

    def show(self, true_robots, obsrv_armors, tracker_info, camera):
        self.screen.fill(Color.BLACK)

        self.show_main_screen(true_robots, obsrv_armors, tracker_info, camera)
        self.show_camera_screen(true_robots, obsrv_armors, tracker_info, camera)
        self.show_info_screen(true_robots, obsrv_armors, tracker_info)

    def show_main_screen(self, true_robots, obsrv_armors, tracker_info, camera):
        main_screen_rect = pg.Rect(
            self.main_screen_center[0] - self.main_screen_width // 2,
            self.main_screen_center[1] - self.main_screen_height // 2,
            self.main_screen_width,
            self.main_screen_height
        )
        pg.draw.rect(self.screen, (20, 20, 20), main_screen_rect)
        pg.draw.rect(self.screen, Color.WHITE, main_screen_rect, 2)

        # 画真实装甲板
        # 也就是真实数据 true data
        for robot in true_robots:
            # 车，装甲板的可视化
            robot_main_screen_pos = world_to_main_screen(
                world_pos=robot.world_pos,
                main_screen_center=self.main_screen_center,
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
        # 也就是 预测数据 pred
        if tracker_info is not None and tracker_info.is_tracking:
            if tracker_info.pred_pos[0] is not None:
                pred_center_main_screen_pos = world_to_main_screen(
                    [
                        tracker_info.pred_pos[0][0],
                        tracker_info.pred_pos[0][1]
                    ],
                    self.main_screen_center,
                    self.world_scale
                )
                pg.draw.circle(self.screen, Color.RED, pred_center_main_screen_pos, 6)

            if len(tracker_info.pred_pos) > 1:
                for armor_id in range(len(tracker_info.pred_pos[1:])):
                    pred_armor_pos = tracker_info.pred_pos[armor_id+1]
                    pred_armor_main_screen_pos = world_to_main_screen(
                        pred_armor_pos,
                        self.main_screen_center,
                        self.world_scale
                    )
                    pg.draw.circle(self.screen, Color.PURPLE, pred_armor_main_screen_pos, 4)

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

    def show_camera_screen(self, true_robots, obsrv_armors, tracker_info, camera):
        camera_screen_rect = pg.Rect(
            self.camera_screen_center[0] - self.camera_screen_width // 2,
            self.camera_screen_center[1] - self.camera_screen_height // 2,
            self.camera_screen_width,
            self.camera_screen_height
        )
        pg.draw.rect(self.screen, (5, 5, 5), camera_screen_rect)
        pg.draw.rect(self.screen, Color.WHITE, camera_screen_rect, 2)

        camera_screen_resolution = (self.camera_screen_width, self.camera_screen_height)

        # 画真实装甲板
        # 也就是真实数据 true data
        for robot in true_robots:
            # 车，装甲板的可视化
            robot_camera_screen_pos = world_to_camera_screen(
                world_pos=robot.world_pos,
                camera=camera,
                camera_screen_center=self.camera_screen_center,
                resolution=camera_screen_resolution
            )
            if robot_camera_screen_pos:
                pg.draw.circle(self.screen, Color.BLUE, robot_camera_screen_pos, 6)

            if robot.armors:
                for armor in robot.armors:
                    armor_camera_screen_pos = world_to_camera_screen(
                        world_pos=armor.world_pos,
                        camera=camera,
                        camera_screen_center=self.camera_screen_center,
                        resolution=camera_screen_resolution
                    )
                    if armor_camera_screen_pos is not None:
                        pg.draw.circle(self.screen, Color.WHITE, armor_camera_screen_pos, 4)

        # 画 加了高斯噪声的装甲板
        # 也就是观测数据 obsrv
        for obsrv_armor in obsrv_armors:
            obsrv_armor_camera_screen_pos = world_to_camera_screen(
                world_pos=obsrv_armor.world_pos,
                camera=camera,
                camera_screen_center=self.camera_screen_center,
                resolution=camera_screen_resolution
            )
            if obsrv_armor_camera_screen_pos is not None:
                pg.draw.circle(self.screen, Color.YELLOW, obsrv_armor_camera_screen_pos, 5)

        # 画 模型导出的数据
        # 也就是 预测数据 pred
        if tracker_info is not None and tracker_info.is_tracking:
            if tracker_info.pred_pos[0] is not None:
                pred_center_camera_screen_pos = world_to_camera_screen(
                    world_pos=tracker_info.pred_pos[0],
                    camera=camera,
                    camera_screen_center=self.camera_screen_center,
                    resolution=camera_screen_resolution
                )
                if pred_center_camera_screen_pos is not None:
                    pg.draw.circle(self.screen, Color.RED, pred_center_camera_screen_pos, 6)

            if len(tracker_info.pred_pos) > 1:
                for armor_id in range(len(tracker_info.pred_pos[1:])):
                    pred_armor_pos = tracker_info.pred_pos[armor_id+1]
                    pred_armor_camera_screen_pos = world_to_camera_screen(
                        world_pos=pred_armor_pos,
                        camera=camera,
                        camera_screen_center=self.camera_screen_center,
                        resolution=camera_screen_resolution
                    )
                    if pred_armor_camera_screen_pos is not None:
                        pg.draw.circle(self.screen, Color.PURPLE, pred_armor_camera_screen_pos, 4)

    def show_info_screen(self, true_robots, obsrv_armors, tracker_info):
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

        for true_robot in true_robots:
            entry_append(
                f"true robot pos: {', '.join(f'{x:.3f}' for x in true_robot.world_pos)}",
                Color.CYAN
            )
            for armor in true_robot.armors:
                entry_append(
                    f"true armor pos: {', '.join(f'{x:.3f}' for x in armor.world_pos)}",
                    Color.GREEN
                )
                # entry_append(
                #     f"true armor rpy: {', '.join(f'{(x+math.pi)/(2*math.pi):.3f}' for x in armor.world_rpy)}",
                #     Color.CYAN
                # )
        # # for obsrv_armor in obsrv_armors:
        # #     entry_append(
        # #         f"obsrv armor rpy: {', '.join(f'{x*(2/math.pi)+(1/2):.3f}' for x in obsrv_armor.world_rpy)}",
        # #         Color.WHITE
        # #     )
        # if len(obsrv_armors) == 1:
        #     entry_append(
        #         f"padding",
        #         Color.WHITE
        #     )

        if tracker_info is not None:
            entry_append(
                f"is tracking: {tracker_info.is_tracking}",
                Color.GREEN
            )
            if tracker_info.pred_pos:
                entry_append(
                    f"pred robot pos: ({', '.join(f'{x:.3f}' for x in tracker_info.pred_pos[0])})" if tracker_info.pred_pos[0] is not None else "pred robot pos: None",
                    Color.WHITE
                )
                if len(tracker_info.pred_pos) > 1:
                    for i in range(1, len(tracker_info.pred_pos)):
                        entry_append(
                            f"pred armor pos: ({tracker_info.pred_pos[i][0]:.3f}, {tracker_info.pred_pos[i][1]:.3f}, {tracker_info.pred_pos[i][2]:.3f})",
                            Color.GREEN
                        )
            if len(tracker_info.state_vecs) >= 1:
                for i in range(len(tracker_info.state_vecs)):
                    entry_append(
                        f"state vec: ({', '.join(f'{x:.3f}' for x in tracker_info.state_vecs[i])})",
                        Color.WHITE
                    )
            # if len(tracker_info.state_vecs) == 3:
            #     # entry_append(
            #     #     f"main model x: ({', '.join(f'{x:.3f}' for x in tracker_info.state_vecs[0])})",
            #     #     Color.WHITE
            #     # )
            #     entry_append(
            #         f"main model x: {tracker_info.state_vecs[0][0]:.3f}",
            #         Color.CYAN
            #     )
            #     entry_append(
            #         f"main model y: {tracker_info.state_vecs[0][1]:.3f}",
            #         Color.CYAN
            #     )
            #     entry_append(
            #         f"main model z: {tracker_info.state_vecs[0][2]:.3f}",
            #         Color.CYAN
            #     )
            #     entry_append(
            #         f"main model v: {tracker_info.state_vecs[0][3]:.3f}",
            #         Color.CYAN
            #     )
            #     entry_append(
            #         f"main model vz: {tracker_info.state_vecs[0][4]:.3f}",
            #         Color.CYAN
            #     )
            #     entry_append(
            #         f"main model angle: {tracker_info.state_vecs[0][5]:.3f}",
            #         Color.CYAN
            #     )
            #     entry_append(
            #         f"main model w: {tracker_info.state_vecs[0][6]:.3f}",
            #         Color.CYAN
            #     )
            #     entry_append(
            #         f"main model a: {tracker_info.state_vecs[0][7]:.3f}",
            #         Color.CYAN
            #     )
            #     entry_append(
            #         f"main model theta: {tracker_info.state_vecs[0][8]:.3f}",
            #         Color.CYAN
            #     )
            #     entry_append(
            #         f"main model omega: {tracker_info.state_vecs[0][9]:.3f}",
            #         Color.CYAN
            #     )
            #     entry_append(
            #         f"main model beta: {tracker_info.state_vecs[0][10]:.3f}",
            #         Color.CYAN
            #     )
            #     entry_append(
            #         f"main model r: {tracker_info.state_vecs[0][11]:.3f}",
            #         Color.CYAN
            #     )
            #
            #     entry_append(
            #         f"center model x: ({', '.join(f'{x:.3f}' for x in tracker_info.state_vecs[1])})",
            #         Color.GREEN
            #     )
            #     entry_append(
            #         f"omega model x: ({', '.join(f'{x:.3f}' for x in tracker_info.state_vecs[2])})",
            #         Color.WHITE
            #     )
            entry_append(
                f"fps: {tracker_info.fps:.3f}",
                Color.GREEN
            )

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











