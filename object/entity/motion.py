import numpy as np
import math

from utils.math_tool import pos_to_tpd


class Motion:
    @staticmethod
    def hrz_osc(state, t, dt):
        ampl = 200.0
        freq = 0.25

        x = (ampl * (1 - math.cos(4 * math.pi * freq * t)) ** 2 / 4) if 0 <= (t % 4 * math.pi) < 2 * math.pi else -(
                    ampl * (1 - math.cos(4 * math.pi * freq * t)) ** 2 / 4)
        vx = (ampl * math.pi * freq * math.sin(2 * math.pi * freq * t) * (1 - math.cos(2 * math.pi * freq * t))) if 0 <= (
                    t % 4 * math.pi) < 2 * math.pi else -(
                    ampl * math.pi * freq * math.sin(2 * math.pi * freq * t) * (1 - math.cos(2 * math.pi * freq * t)))

        state.pos[0] = x
        state.vel[0] = vx
        state.tpd = pos_to_tpd(state.pos)

    @staticmethod
    def go_right(state, t, dt):
        state.vel = np.array([5., 0., 0.])
        state.pos += state.vel * dt
        state.tpd = pos_to_tpd(state.pos)

    @staticmethod
    def rotation(state, t, dt):
        state.omg = np.array([0., 0., 0.01])
        state.rpy += state.omg * dt
        state.tpd = pos_to_tpd(state.pos)






