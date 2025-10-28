import numpy as np
import math

from utils.math_tool import pos_to_tpd, limit_rad

STEP = 80.


class Motion:
    @staticmethod
    def up_down_osc(state, t, dt):
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
    def left_right_osc(state, t, dt):
        ampl = 180.0
        freq = 0.25

        y = (ampl * (1 - math.cos(4 * math.pi * freq * t)) ** 2 / 4) if 0 <= (t % 4 * math.pi) < 2 * math.pi else -(
                    ampl * (1 - math.cos(4 * math.pi * freq * t)) ** 2 / 4)
        vy = (ampl * math.pi * freq * math.sin(2 * math.pi * freq * t) * (1 - math.cos(2 * math.pi * freq * t))) if 0 <= (
                    t % 4 * math.pi) < 2 * math.pi else -(
                    ampl * math.pi * freq * math.sin(2 * math.pi * freq * t) * (1 - math.cos(2 * math.pi * freq * t)))

        state.pos[1] = y
        state.vel[1] = vy
        state.tpd = pos_to_tpd(state.pos)

    @staticmethod
    def diag_osc(state, t, dt):
        ampl = 140.0
        freq = 0.25

        sgl_pos = (ampl * (1 - math.cos(4 * math.pi * freq * t)) ** 2 / 4) if 0 <= (t % 4 * math.pi) < 2 * math.pi else -(
                    ampl * (1 - math.cos(4 * math.pi * freq * t)) ** 2 / 4)
        sgl_vel = (ampl * math.pi * freq * math.sin(2 * math.pi * freq * t) * (1 - math.cos(2 * math.pi * freq * t))) if 0 <= (
                    t % 4 * math.pi) < 2 * math.pi else -(
                    ampl * math.pi * freq * math.sin(2 * math.pi * freq * t) * (1 - math.cos(2 * math.pi * freq * t)))

        state.pos[0] = sgl_pos
        state.pos[1] = sgl_pos
        state.vel[0] = sgl_vel
        state.vel[1] = sgl_vel
        state.tpd = pos_to_tpd(state.pos)

    @staticmethod
    def up_right_osc_rotate(state, t, dt):
        ampl = 200.0
        freq = 0.25

        x = (ampl * (1 - math.cos(4 * math.pi * freq * t)) ** 2 / 4) if 0 <= (t % 4 * math.pi) < 2 * math.pi else -(
                ampl * (1 - math.cos(4 * math.pi * freq * t)) ** 2 / 4)
        vx = (ampl * math.pi * freq * math.sin(2 * math.pi * freq * t) * (
                    1 - math.cos(2 * math.pi * freq * t))) if 0 <= (
                t % 4 * math.pi) < 2 * math.pi else -(
                ampl * math.pi * freq * math.sin(2 * math.pi * freq * t) * (1 - math.cos(2 * math.pi * freq * t)))

        state.pos[0] = x
        state.vel[0] = vx

        state.omg = np.array([0., 0., 2 * math.pi])
        state.rpy += state.omg * dt
        state.rpy = limit_rad(state.rpy)

        state.tpd = pos_to_tpd(state.pos)

    @staticmethod
    def left_right_osc_rotate(state, t, dt):
        ampl = 180.0
        freq = 0.25

        y = (ampl * (1 - math.cos(4 * math.pi * freq * t)) ** 2 / 4) if 0 <= (t % 4 * math.pi) < 2 * math.pi else -(
                    ampl * (1 - math.cos(4 * math.pi * freq * t)) ** 2 / 4)
        vy = (ampl * math.pi * freq * math.sin(2 * math.pi * freq * t) * (1 - math.cos(2 * math.pi * freq * t))) if 0 <= (
                    t % 4 * math.pi) < 2 * math.pi else -(
                    ampl * math.pi * freq * math.sin(2 * math.pi * freq * t) * (1 - math.cos(2 * math.pi * freq * t)))

        state.pos[1] = y
        state.vel[1] = vy

        state.omg = np.array([0., 0., 2 * math.pi])
        state.rpy += state.omg * dt
        state.rpy = limit_rad(state.rpy)

        state.tpd = pos_to_tpd(state.pos)

    @staticmethod
    def diag_osc_rotate(state, t, dt):
        ampl = 140.0
        freq = 0.25

        sgl_pos = (ampl * (1 - math.cos(4 * math.pi * freq * t)) ** 2 / 4) if 0 <= (t % 4 * math.pi) < 2 * math.pi else -(
                    ampl * (1 - math.cos(4 * math.pi * freq * t)) ** 2 / 4)
        sgl_vel = (ampl * math.pi * freq * math.sin(2 * math.pi * freq * t) * (1 - math.cos(2 * math.pi * freq * t))) if 0 <= (
                    t % 4 * math.pi) < 2 * math.pi else -(
                    ampl * math.pi * freq * math.sin(2 * math.pi * freq * t) * (1 - math.cos(2 * math.pi * freq * t)))

        state.pos[0] = sgl_pos
        state.pos[1] = sgl_pos
        state.vel[0] = sgl_vel
        state.vel[1] = sgl_vel

        state.omg = np.array([0., 0., 2 * math.pi])
        state.rpy += state.omg * dt
        state.rpy = limit_rad(state.rpy)

        state.tpd = pos_to_tpd(state.pos)

    @staticmethod
    def rotate_anticlockwise(state, t, dt):
        state.omg = np.array([0., 0., -2.])
        state.rpy += state.omg * dt
        state.rpy = limit_rad(state.rpy)
        state.tpd = pos_to_tpd(state.pos)

    @staticmethod
    def rotate_clockwise(state, t, dt):
        state.omg = np.array([0., 0., 2.])
        state.rpy += state.omg * dt
        state.rpy = limit_rad(state.rpy)
        state.tpd = pos_to_tpd(state.pos)

    @staticmethod
    def top_rotate_anticlockwise(state, t, dt):
        state.omg = np.array([0., 0., -10.])
        state.rpy += state.omg * dt
        state.rpy = limit_rad(state.rpy)
        state.tpd = pos_to_tpd(state.pos)

    @staticmethod
    def top_rotate_clockwise(state, t, dt):
        state.omg = np.array([0., 0., 10.])
        state.rpy += state.omg * dt
        state.rpy = limit_rad(state.rpy)
        state.tpd = pos_to_tpd(state.pos)

    @staticmethod
    def go_up(state, t, dt):
        state.vel = np.array([STEP, 0., 0.])
        state.pos += state.vel * dt
        state.tpd = pos_to_tpd(state.pos)

    @staticmethod
    def go_down(state, t, dt):
        state.vel = np.array([-STEP, 0., 0.])
        state.pos += state.vel * dt
        state.tpd = pos_to_tpd(state.pos)

    @staticmethod
    def go_left(state, t, dt):
        state.vel = np.array([0., -STEP, 0.])
        state.pos += state.vel * dt
        state.tpd = pos_to_tpd(state.pos)

    @staticmethod
    def go_right(state, t, dt):
        state.vel = np.array([0., STEP, 0.])
        state.pos += state.vel * dt
        state.tpd = pos_to_tpd(state.pos)

    @staticmethod
    def go_up_rotate(state, t, dt):
        state.vel = np.array([STEP, 0., 0.])
        state.pos += state.vel * dt
        state.tpd = pos_to_tpd(state.pos)

        state.omg = np.array([0., 0., 2 * math.pi])
        state.rpy += state.omg * dt
        state.rpy = limit_rad(state.rpy)

    @staticmethod
    def go_down_rotate(state, t, dt):
        state.vel = np.array([-STEP, 0., 0.])
        state.pos += state.vel * dt
        state.tpd = pos_to_tpd(state.pos)

        state.omg = np.array([0., 0., 2 * math.pi])
        state.rpy += state.omg * dt
        state.rpy = limit_rad(state.rpy)

    @staticmethod
    def go_left_rotate(state, t, dt):
        state.vel = np.array([0., -STEP, 0.])
        state.pos += state.vel * dt
        state.tpd = pos_to_tpd(state.pos)

        state.omg = np.array([0., 0., 2 * math.pi])
        state.rpy += state.omg * dt
        state.rpy = limit_rad(state.rpy)

    @staticmethod
    def go_right_rotate(state, t, dt):
        state.vel = np.array([0., STEP, 0.])
        state.pos += state.vel * dt
        state.tpd = pos_to_tpd(state.pos)

        state.omg = np.array([0., 0., 2 * math.pi])
        state.rpy += state.omg * dt
        state.rpy = limit_rad(state.rpy)

    @staticmethod
    def stay_inclined(state, t, dt):
        state.rpy = np.array([0., 0., math.pi / 4])

    @staticmethod
    def stay_still(state, t, dt):
        pass






