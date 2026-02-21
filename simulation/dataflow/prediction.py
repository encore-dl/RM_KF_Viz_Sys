from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class Prediction:
    """跟踪器的预测结果"""
    center: np.ndarray          # 机器人中心位置 (3,)
    armors: List[np.ndarray]    # 各个装甲板位置列表 (长度等于装甲板数量)
    timestamp: float            # 预测对应的时间戳
    is_tracking: bool           # 是否正在跟踪
    fps: float
    state_vector: Optional[np.ndarray] = None


