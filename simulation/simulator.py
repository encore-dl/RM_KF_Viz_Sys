import pygame.time as pgtime

from simulation.managers.entity_manager.robot_manager import RobotManager
from simulation.managers.entity_manager.motion_manager import MotionManager
from simulation.managers.system_manager.sensor_manager import SensorManager
from simulation.managers.system_manager.visualization_manager import VisualizationManager
from simulation.managers.system_manager.tracker_manager import TrackerManager
from simulation.managers.system_manager.bullet_manager import BulletManager
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

        self.motion_manager = MotionManager()
        self.robot_manager = RobotManager(self.motion_manager)
        self.bullet_manager = BulletManager(self.robot_manager)
        self.sensor_manager = SensorManager(self.robot_manager)
        self.visualization_manager = VisualizationManager(SCREEN_WIDTH, SCREEN_HEIGHT, self.robot_manager)
        self.tracker_manager = TrackerManager()

        event_bus.subscribe('fire', self._on_fire_command)

    def run_simulator(self):
        self.update()
        self.sensor_manager.get_obs()
        self.bullet_manager.update(self.dt)
        self.visualization_manager.show(self.bullet_manager.bullets)

    def update(self):
        curr_t = pgtime.get_ticks()
        self.dt = (curr_t - self.last_t) / 1000.
        self.last_t = curr_t

        self.motion_manager.update(self.dt, curr_t)

        self.clock.tick(self.simulator_fps)

    def select_next_robot(self):
        self.robot_manager.next_robot()
        self.selected_entity = self.robot_manager.selected_robot

    def _on_fire_command(self, data):
        pos = data['pos']
        vel = data['vel']
        self.bullet_manager.fire_bullet(pos, vel)