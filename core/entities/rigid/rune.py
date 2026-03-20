import numpy as np
import copy

from core.entities.rigid.rigid import Rigid
from config.config_manager import cfg_mgr


class Rune(Rigid):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cfg = cfg_mgr.rune_cfg
        self.leaf_count = cfg.leaf_count
        self.radius = cfg.radius













