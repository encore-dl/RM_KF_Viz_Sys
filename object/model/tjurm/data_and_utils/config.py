from dataclasses import dataclass

from object.model.tjurm.data_and_utils.slide_integrator import SlideIntegrator

@dataclass
class TJURMConfig:
    """TJURM模型配置参数"""
    # TrackQueue参数
    track_count: int = 10
    track_distance: float = 0.2
    track_delay: float = 0.5
    track_fire_interval: float = 0.05
    track_fire_high_delay: float = 0.02

    # Antitop参数
    antitop_min_r: float = 0.
    antitop_max_r: float = 5
    antitop_armor_num: int = 4
    antitop_fire_retention: float = 5.0
    antitop_fire_update: int = 100
    antitop_fire_delay: float = 1.

    # 切换阈值
    track_to_antitop: float = 1.
    antitop_to_track: float = 0.6
    armor_to_center: float = 5.
    center_to_armor: float = 7.

    slide_integrator = SlideIntegrator(2000, 2.0)

    rotate_delay = 0.05


