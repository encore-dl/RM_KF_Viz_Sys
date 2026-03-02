from dataclasses import dataclass
from typing import List
import numpy as np

from core.entities.property.robot_type import RobotType


@dataclass
class ArmorObservation:
    armor_id: int
    robot_type: RobotType
    rel_pos: np.ndarray          # (3,)
    rel_rpy: np.ndarray          # (3,)
    armor_size: str                # 'large' 或 'small'
    pixel_points: np.ndarray = None   # 4 个角点的像素坐标 (4,2)


@dataclass
class Observation:
    obs_armors: List[ArmorObservation]
    timestamp: float          # 观测时间戳（秒）






