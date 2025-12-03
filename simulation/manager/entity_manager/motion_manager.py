from typing import Callable
import numpy as np
import math
from dataclasses import dataclass

from object.entity.motion import Motion
from utils.math_tool import pos_to_tpd, limit_rad, get_euler_rotate_matrix


@dataclass
class MotionState:
    pos: np.ndarray
    vel: np.ndarray
    rpy: np.ndarray
    omg: np.ndarray
    tpd: np.ndarray


class MotionManager:
    def __init__(self):
        self.entity_register = {}
        self.init_state = {}

        self.motion = Motion()

    def _record_init_state(self, entity):
        if hasattr(entity, 'armors'):
            self.init_state[entity] = {
                'armor_offs': [
                    armor.world_pos - entity.world_pos for armor in entity.armors
                ]
            }

    def set_motion(self, entity, motion_func: Callable):
        if entity is None:
            return
        self.entity_register[entity] = motion_func
        if (entity not in self.init_state) or \
           (entity in self.init_state and motion_func != self.entity_register[entity]):
            self._record_init_state(entity)

    def update(self, dt, t):
        for entity, motion_func in self.entity_register.items():
            cur_state = MotionState(
                pos=entity.world_pos.copy(),
                vel=entity.world_vel.copy(),
                rpy=entity.world_rpy.copy(),
                omg=entity.world_omg.copy(),
                tpd=entity.world_tpd.copy()
            )

            # 使用替身state来先进行更新
            motion_func(cur_state, t, dt)

            self._apply_motion_state(entity, cur_state)

            if hasattr(entity, 'armors'):
                self._update_armors(entity, cur_state)

    @staticmethod
    def _apply_motion_state(entity, state):
        entity.world_pos = state.pos
        entity.world_vel = state.vel
        entity.world_rpy = state.rpy
        entity.world_omg = state.omg
        entity.world_tpd = state.tpd

        if hasattr(entity, 'auto_aiming'):
            entity.world_rpy[2] = limit_rad(entity.world_rpy[2])
            entity.world_rpy[1] = np.clip(entity.world_rpy[1], -math.pi / 2, math.pi / 2)

    def _update_armors(self, robot, state):
        if robot not in self.init_state:
            return

        armor_offs = self.init_state[robot]['armor_offs']
        rotate_mat = get_euler_rotate_matrix(state.rpy)

        for i, armor in enumerate(robot.armors):
            rotate_offs = rotate_mat @ armor_offs[i]

            armor.world_pos = state.pos + rotate_offs
            armor.world_vel = state.vel + np.cross(state.omg, rotate_offs)
            armor.world_rpy = state.rpy.copy()
            armor.world_rpy[2] = limit_rad(state.rpy[2] + (armor.armor_id * 2 * math.pi / robot.armor_count))
            armor.world_omg = state.omg.copy()
            armor.world_tpd = pos_to_tpd(armor.world_pos)





