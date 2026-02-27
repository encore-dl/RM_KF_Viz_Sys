import numpy as np
from core.entities.rigid.rigid import Rigid


class Muzzle(Rigid):
    def __init__(self, mount_pos=np.array([0.2, 0, 0]), mount_rpy=np.zeros(3), **kwargs):
        super().__init__(**kwargs)
        self.mount_pos = mount_pos.copy()
        self.mount_rpy = mount_rpy.copy()