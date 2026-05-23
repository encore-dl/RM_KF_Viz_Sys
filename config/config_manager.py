import yaml
import os
from dataclasses import dataclass

from core.entities.property.robot_type import RobotType


@dataclass
class RobotTypeConfig:
    length: float
    width: float
    armor_count: int
    high_height: float
    low_height: float
    armor_size: str
    light_bar_interval: float
    light_bar_length: float


@dataclass
class OutpostConfig:
    radius: float
    armor_count: int
    base_height: float
    height_step: float
    rotate_speed: float
    armor_size: str
    light_bar_interval: float
    light_bar_length: float


@dataclass
class RuneConfig:
    leaf_count: int
    radius: float
    height: float


@dataclass
class SimulatorConfig:
    screen_width: int
    screen_height: int
    sim_fps: int
    sensor_fps: int
    world_scale: float
    pos_noise_sigma: float
    rpy_noise_sigma: float
    pixel_noise_sigma: float


class ConfigManager:
    def __init__(self, config_dir='config'):
        self.robots_cfg = {}
        self.rune_cfg = {}
        self.sim_cfg = None

        robot_path = os.path.join(config_dir, 'robots.yaml')
        with open(robot_path, 'r') as f:
            robot_data = yaml.safe_load(f)
        for name, params in robot_data['Robot'].items():
            self.robots_cfg[name] = RobotTypeConfig(**params)
        self.outpost_cfg = OutpostConfig(**robot_data['Outpost'])
        self.rune_cfg = RuneConfig(**robot_data['Rune'])

        sim_path = os.path.join(config_dir, 'simulator.yaml')
        with open(sim_path, 'r') as f:
            sim_data = yaml.safe_load(f)
        self.sim_cfg = SimulatorConfig(**sim_data)

    def get_robot_config(self, robot_type) -> RobotTypeConfig:
        robot_name = RobotType.get_name(robot_type)
        return self.robots_cfg[robot_name]

    def get_simulator_config(self):
        return self.sim_cfg

    def get_outpost_config(self):
        return self.outpost_cfg

    def get_rune_config(self):
        return self.rune_cfg


cfg_mgr = ConfigManager()

