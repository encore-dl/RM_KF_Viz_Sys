import pygame.time as pgtime

from config.config_manager import cfg_mgr
from simulation.managers.robot_manager import RobotManager
from simulation.managers.motion_manager import MotionManager
from simulation.managers.sensor_manager import SensorManager
from simulation.managers.visualization_manager import VisualizationManager
from simulation.managers.bullet_manager import BulletManager


class Simulator:
    def __init__(self):
        cfg = cfg_mgr.sim_cfg

        self.clock = pgtime.Clock()
        self.last_t = pgtime.get_ticks()
        self.dt = 0.
        self.sim_fps = cfg.sim_fps

        self.selected_entity = None

        self.motion_manager = MotionManager()
        self.robot_manager = RobotManager(self.motion_manager)
        self.bullet_manager = BulletManager(self.robot_manager)
        self.sensor_manager = SensorManager(self.robot_manager)
        self.visualization_manager = VisualizationManager(self.robot_manager, self.bullet_manager)

    def run_simulator(self):
        self.update()
        self.sensor_manager.get_obs()
        self.visualization_manager.show()

    def update(self):
        curr_t = pgtime.get_ticks()
        self.dt = (curr_t - self.last_t) / 1000.
        self.last_t = curr_t

        self.motion_manager.update(self.dt, curr_t)
        self.bullet_manager.update(self.dt)

        self.clock.tick(self.sim_fps)




