import math
import random

import numpy as np

from core.entities.rigid.rigid import Rigid
from config.config_manager import cfg_mgr
from core.algorithms.math.angle import limit_rad


class Rune(Rigid):
    def __init__(self, world_pos=None, **kwargs):
        cfg = cfg_mgr.rune_cfg
        default_pos = np.array([0.0, 0.0, cfg.height])
        if world_pos is None:
            world_pos = default_pos.copy()
        super().__init__(world_pos=world_pos, **kwargs)

        self.leaf_count = cfg.leaf_count
        self.radius = cfg.radius
        self.height = cfg.height

        self.is_big_rune = False
        self.is_activated = False
        self.rotate_dir = random.choice([-1, 1])

        self.small_speed = math.pi / 3.
        self.big_base_speed = math.pi / 3.

        self.activation_params = None
        self.activation_timer = 0.

        self.leaf_centers = [np.zeros(3) for _ in range(self.leaf_count)]
        self.update_leaves()

    def set_mode(self, is_big_rune: bool):
        self.is_big_rune = is_big_rune
        if not self.is_big_rune:
            self.is_activated = False

    def activate(self):
        if not self.is_big_rune:
            return
        a = random.uniform(0.780, 1.045)
        omg = random.uniform(1.884, 2.000)
        b = 2.090 - a
        self.activation_params = {'a': a, 'omg': omg, 'b': b}
        self.activation_timer = 0.0
        self.is_activated = True

    def deactivate(self):
        self.is_activated = False

    def update_rune(self, dt):
        if self.is_big_rune and self.is_activated:
            t = self.activation_timer
            a = self.activation_params['a']
            omg = self.activation_params['omg']
            b = self.activation_params['b']
            speed = a * math.sin(omg * t) + b
            self.activation_timer += dt
        else:
            speed = self.big_base_speed if self.is_big_rune else self.small_speed

        self.world_omg[1] = self.rotate_dir * speed
        self.world_rpy[1] += self.world_omg[1] * dt
        self.world_rpy[1] = limit_rad(self.world_rpy[1])

        self.update_leaves()

    def update_leaves(self):
        base_angle = self.world_rpy[1]
        for i in range(self.leaf_count):
            angle = base_angle + i * 2. * math.pi / self.leaf_count
            x = self.world_pos[0] + self.radius * math.cos(angle)
            y = self.world_pos[1]
            z = self.world_pos[2] + self.radius * math.sin(angle)
            self.leaf_centers[i] = np.array([x, y, z])












