from enum import Enum
from dataclasses import dataclass

from object.entity.motion import Motion


class MotionManager:
    def __init__(self):
        self.motion = Motion()
        self._entity = None

        self.auto_pos_step = None
        self.auto_rpy_step = None

        self.manual = False

    def change_motion(self, entity, do_motion, t=None, dt=None):
        self._entity = entity
        self.motion.change_entity(self._entity)
        do_motion(self.auto_pos_step, self.auto_rpy_step, t, dt)

    def set_auto_step(self, auto_pos_step, auto_rpy_step):
        self.auto_pos_step = auto_pos_step
        self.auto_rpy_step = auto_rpy_step






