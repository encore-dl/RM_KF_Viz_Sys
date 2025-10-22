import math
import numpy as np

from utils.math_tool import limit_rad


class Motion:
    def __init__(self):
        self._entity = None

    def change_entity(self, entity):
        self._entity = entity
    
    def hrz_osc(self, auto_pos_step=None, auto_rpy_step=None, t=None, dt=None):
        ampl = 200.0
        freq = 0.25
        
        x = (ampl * (1 - math.cos(4 * math.pi * freq * t)) ** 2 / 4) if 0 <= (t % 4 * math.pi) < 2 * math.pi else -(
                    ampl * (1 - math.cos(4 * math.pi * freq * t)) ** 2 / 4)
        vx = (ampl * math.pi * freq * math.sin(2 * math.pi * freq * t) * (1 - math.cos(2 * math.pi * freq * t))) if 0 <= (
                    t % 4 * math.pi) < 2 * math.pi else -(
                    ampl * math.pi * freq * math.sin(2 * math.pi * freq * t) * (1 - math.cos(2 * math.pi * freq * t)))

        self._entity.world_pos[0] = x
        self._entity.world_vel[0] = vx

    def go_up(self, auto_pos_step=None, auto_rpy_step=None, t=None, dt=None):
        self._entity.world_pos += self._entity.entities.world_vel * dt

    def go_down(self, auto_pos_step=None, auto_rpy_step=None, t=None, dt=None):
        pass

    def go_left(self, auto_pos_step=None, auto_rpy_step=None, t=None, dt=None):
        pass

    def go_right(self, auto_pos_step=None, auto_rpy_step=None, t=None, dt=None):
        pass

    def camera_auto_motion(self, auto_pos_step=None, auto_rpy_step=None, t=None, dt=None):
        if auto_pos_step is not None:
            self._entity.world_pos[0] = self._entity.world_pos[0] + auto_pos_step[0] * dt
            self._entity.world_pos[1] = self._entity.world_pos[1] + auto_pos_step[1] * dt
            self._entity.world_pos[2] = self._entity.world_pos[2] + auto_pos_step[2] * dt
        if auto_rpy_step is not None:
            self._entity.world_rpy[2] += auto_rpy_step[2] * dt
            self._entity.world_rpy[1] += auto_rpy_step[1] * dt

        self._entity.world_rpy[2] = limit_rad(self._entity.world_rpy[2])
        self._entity.world_rpy[1] = np.clip(self._entity.world_rpy[1], -math.pi / 2, math.pi / 2)

    def auto_motion(self):
        pass























