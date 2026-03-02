from dataclasses import dataclass
from simulation.dataflow.observation import ArmorObservation
from typing import List


@dataclass
class Tracking:
    obs_armors: List[ArmorObservation]
    input_dt: float
    timestamp: float          # 观测时间戳（秒）
