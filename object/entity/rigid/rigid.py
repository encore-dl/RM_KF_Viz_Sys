import numpy as np
from typing import Optional


class Rigid:
    def __init__(self,
                 world_pos: Optional[np.ndarray] = None,  # 世界坐标
                 world_vel: Optional[np.ndarray] = None,  # 世界速度
                 world_acc: Optional[np.ndarray] = None,  # 世界加速度
                 world_tpd: Optional[np.ndarray] = None,  # 世界球坐标
                 world_rpy: Optional[np.ndarray] = None,  # 世界朝向角
                 world_omg: Optional[np.ndarray] = None,  # 世界角速度
                 world_alp: Optional[np.ndarray] = None,  # 世界角加速度
                 ):
        self.world_pos = world_pos if world_pos is not None else np.array([0., 0., 0.])
        self.world_vel = world_vel if world_vel is not None else np.array([0., 0., 0.])
        self.world_acc = world_acc if world_acc is not None else np.array([0., 0., 0.])
        self.world_tpd = world_tpd if world_tpd is not None else np.array([0., 0., 0.])
        self.world_rpy = world_rpy if world_rpy is not None else np.array([0., 0., 0.])
        self.world_omg = world_omg if world_omg is not None else np.array([0., 0., 0.])
        self.world_alp = world_alp if world_alp is not None else np.array([0., 0., 0.])


