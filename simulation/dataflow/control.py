from dataclasses import dataclass, field
import numpy as np


@dataclass
class ControlCommand:
    entity_id: int           # 被控实体，可以是 Robot 或 Camera 对象（或 id）
    target_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    target_omega: np.ndarray = field(default_factory=lambda: np.zeros(3))


