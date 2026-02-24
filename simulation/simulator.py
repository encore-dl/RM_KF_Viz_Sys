import pygame.time as pgtime
import numpy as np
import time

from simulation.managers.entity_manager.robot_manager import RobotManager
from simulation.managers.entity_manager.sensor_manager import SensorManager
from simulation.managers.entity_manager.motion_manager import MotionManager
from simulation.managers.system_manager.visualization_manager import VisualizationManager
from simulation.managers.system_manager.tracker_manager import TrackerManager

from core.entities.bullet import Bullet
from simulation.event_bus import event_bus

SCREEN_WIDTH = 1500
SCREEN_HEIGHT = 840


class Simulator:
    def __init__(self):
        self.clock = pgtime.Clock()
        self.dt = 0.
        self.last_t = pgtime.get_ticks()
        self.simulator_fps = 500.

        self.selected_entity = None

        self.camera_manager = SensorManager()
        self.robot_manager = RobotManager(self.camera_manager.camera)
        self.motion_manager = MotionManager()
        self.visualization_manager = VisualizationManager(SCREEN_WIDTH, SCREEN_HEIGHT)
        self.tracker_manager = TrackerManager()

        self.bullets = []  # 子弹列表
        self.last_alp = 0.
        self.ctrl_accum = 0.

        # 订阅控制指令
        event_bus.subscribe('gimbal_command', self._on_gimbal_command)
        event_bus.subscribe('fire', self._on_fire)

    def run_simulator(self):
        self.update()

        # 生产被观测的数据，实际上只有被观测的装甲板
        self.camera_manager.get_obs(self.robot_manager.robots)

        self._update_bullets()

        self.visualization_manager.show(
            self.robot_manager.robots,
            self.camera_manager.camera,
            self.bullets
        )

        # self.draw_psi_d_graph()

    def update(self):
        curr_t = pgtime.get_ticks()
        self.dt = (curr_t - self.last_t) / 1000.
        self.last_t = curr_t

        self.motion_manager.update(self.dt, curr_t)

        # 相机自瞄
        if self.camera_manager.camera.auto_aiming:
            self.ctrl_accum += self.dt
            while self.ctrl_accum >= 0.01:
                self.camera_manager.camera.apply_control(self.last_alp, 0.01)
                self.ctrl_accum -= 0.01

        self.clock.tick(self.simulator_fps)

    def select_entity(self, selected_type, entity_number=None):
        if selected_type == 'robot':
            if entity_number is not None and 0 <= entity_number < len(self.robot_manager.robots):
                self.selected_entity = self.robot_manager.robots[entity_number]
            else:
                pass
        elif selected_type == 'camera':
            self.selected_entity = self.camera_manager.camera

    def _on_gimbal_command(self, data):
        """接收云台控制指令"""
        self.last_alp = data['alpha']

    def _on_fire(self, data):
        """发射子弹"""
        pos = data['pos']
        vel = data['vel']
        self.fire_bullet(pos, vel)

    def _update_bullets(self):
        """更新所有子弹位置并检查碰撞"""
        for bullet in self.bullets[:]:  # 遍历副本
            bullet.update(self.dt)
            # 检查是否超出范围或时间过长
            if np.linalg.norm(bullet.pos) > 50 or (time.time() - bullet.birth_time) > 5:
                bullet.active = False
            # 检查与所有装甲板的碰撞
            if bullet.active:
                for robot in self.robot_manager.robots:
                    for armor in robot.armors:
                        if bullet.check_collision(armor):
                            bullet.active = False
                            bullet.hit = True
                            bullet.hit_pos = bullet.pos.copy()
                            bullet.hit_target = (robot.robot_type, armor.armor_id)
                            break
            if not bullet.active:
                self.bullets.remove(bullet)

    def fire_bullet(self, pos, vel):
        """外部调用发射子弹"""
        bullet = Bullet(pos, vel)
        self.bullets.append(bullet)












