import pygame.time as pgtime
import numpy as np
import time

from simulation.managers.entity_manager.robot_manager import RobotManager
from simulation.managers.entity_manager.motion_manager import MotionManager
from simulation.managers.entity_manager.camera_manager import CameraManager
from simulation.managers.system_manager.sensor_manager import SensorManager
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

        self.camera_manager = CameraManager()
        self.camera_manager.create_camera()  # 默认相机

        self.sensor_manager = SensorManager(self.camera_manager)
        self.robot_manager = RobotManager()
        self.motion_manager = MotionManager()
        self.visualization_manager = VisualizationManager(SCREEN_WIDTH, SCREEN_HEIGHT)
        self.tracker_manager = TrackerManager()

        self.bullets = []
        self.last_alp_yaw = 0.
        self.last_alp_pitch = 0.
        self.ctrl_accum = 0.

        event_bus.subscribe('gimbal_yaw', self._on_gimbal_yaw)
        event_bus.subscribe('gimbal_pitch', self._on_gimbal_pitch)
        event_bus.subscribe('fire', self._on_fire_command)

    def run_simulator(self):
        self.update()
        self.sensor_manager.get_obs(self.robot_manager.robots)
        self._update_bullets()
        self.visualization_manager.show(
            self.robot_manager.robots,
            self.camera_manager.cameras,
            self.camera_manager.selected_camera,
            self.bullets
        )

    def update(self):
        curr_t = pgtime.get_ticks()
        self.dt = (curr_t - self.last_t) / 1000.
        self.last_t = curr_t

        self.motion_manager.update(self.dt, curr_t)

        if self.camera_manager.selected_camera and self.camera_manager.selected_camera.auto_aiming:
            self.ctrl_accum += self.dt
            while self.ctrl_accum >= 0.01:
                self.camera_manager.selected_camera.apply_control(self.last_alp_yaw, self.last_alp_pitch, 0.01)
                self.ctrl_accum -= 0.01

        self.clock.tick(self.simulator_fps)

    def select_entity(self, selected_type, entity_number=None):
        if selected_type == 'robot':
            if entity_number is not None and 0 <= entity_number < len(self.robot_manager.robots):
                self.selected_entity = self.robot_manager.robots[entity_number]
        elif selected_type == 'camera':
            if entity_number is not None and 0 <= entity_number < len(self.camera_manager.cameras):
                self.selected_entity = self.camera_manager.cameras[entity_number]
                self.camera_manager.selected_camera = self.selected_entity
            elif self.camera_manager.cameras:
                self.selected_entity = self.camera_manager.cameras[0]
                self.camera_manager.selected_camera = self.selected_entity

    def select_next_robot(self):
        if not self.robot_manager.robots:
            self.selected_entity = None
            return
        if self.selected_entity in self.robot_manager.robots:
            idx = self.robot_manager.robots.index(self.selected_entity)
            idx = (idx + 1) % len(self.robot_manager.robots)
            self.selected_entity = self.robot_manager.robots[idx]
        else:
            self.selected_entity = self.robot_manager.robots[0]

    def select_next_camera(self):
        if not self.camera_manager.cameras:
            self.selected_entity = None
            return
        if self.selected_entity in self.camera_manager.cameras:
            idx = self.camera_manager.cameras.index(self.selected_entity)
            idx = (idx + 1) % len(self.camera_manager.cameras)
            self.selected_entity = self.camera_manager.cameras[idx]
            self.camera_manager.selected_camera = self.selected_entity
        else:
            self.selected_entity = self.camera_manager.cameras[0]
            self.camera_manager.selected_camera = self.selected_entity

    def _on_gimbal_yaw(self, data):
        self.last_alp_yaw = data['alpha']

    def _on_gimbal_pitch(self, data):
        self.last_alp_pitch = data['alpha']

    def _on_fire_command(self, data):
        pos = data['pos']
        vel = data['vel']
        self.fire_bullet(pos, vel)

    def _update_bullets(self):
        curr_t = time.time()
        for bullet in self.bullets[:]:
            if bullet.hit:
                if curr_t - bullet.hit_time > 0.5:
                    self.bullets.remove(bullet)
                continue
            prev_pos = bullet.pos.copy()
            bullet.update(self.dt)
            curr_pos = bullet.pos
            if np.linalg.norm(curr_pos) > 50 or (time.time() - bullet.birth_time) > 5:
                bullet.active = False
                self.bullets.remove(bullet)
                continue
            hit_occurred = False
            for robot in self.robot_manager.robots:
                for armor in robot.armors:
                    hit, hit_pos = bullet.check_segment_collision(prev_pos, curr_pos, armor)
                    if hit:
                        bullet.hit = True
                        bullet.hit_pos = hit_pos
                        bullet.hit_target = (robot.robot_type, armor.armor_id)
                        bullet.hit_time = curr_t
                        hit_occurred = True
                        break
                if hit_occurred:
                    break

    def fire_bullet(self, pos, vel):
        bullet = Bullet(pos, vel)
        self.bullets.append(bullet)