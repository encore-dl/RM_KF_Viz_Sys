import numpy as np
import math
from dataclasses import dataclass, field


@dataclass
class MotionConfig:
    T_STEP = 3.0  # 加速度/速度增量
    R_STEP_SLOW = 2.0
    R_STEP_FAST = 10.0


@dataclass
class MotionState:
    pos: np.ndarray
    vel: np.ndarray
    tpd: np.ndarray
    rpy: np.ndarray
    omg: np.ndarray
    acc: np.ndarray = field(default_factory=lambda: np.zeros(3))
    alp: np.ndarray = field(default_factory=lambda: np.zeros(3))

    tar_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tar_omg: np.ndarray = field(default_factory=lambda: np.zeros(3))

    mass: float = 1.
    fric_coef: float = 0.5
    motor_gain: float = 6.
    max_force: float = 20.
    max_speed: float = 10.
    max_rotate_speed: float = 20.


class Motion:
    # === 平移类：叠加目标速度 ===

    @staticmethod
    def go_left(state, t, dt):
        state.tar_vel[0] -= MotionConfig.T_STEP

    @staticmethod
    def go_right(state, t, dt):
        state.tar_vel[0] += MotionConfig.T_STEP

    @staticmethod
    def go_up(state, t, dt):
        state.tar_vel[1] += MotionConfig.T_STEP

    @staticmethod
    def go_down(state, t, dt):
        state.tar_vel[1] -= MotionConfig.T_STEP

    @staticmethod
    def ascend(state, t, dt):
        state.tar_vel[2] += MotionConfig.T_STEP

    @staticmethod
    def descend(state, t, dt):
        state.tar_vel[2] -= MotionConfig.T_STEP

    # === 旋转类：叠加目标角速度 ===

    @staticmethod
    def rotate_anticlockwise(state, t, dt):
        state.tar_omg[2] += MotionConfig.R_STEP_SLOW

    @staticmethod
    def rotate_clockwise(state, t, dt):
        state.tar_omg[2] -= MotionConfig.R_STEP_SLOW

    @staticmethod
    def top_rotate_anticlockwise(state, t, dt):
        state.tar_omg[2] += MotionConfig.R_STEP_FAST

    @staticmethod
    def top_rotate_clockwise(state, t, dt):
        state.tar_omg[2] -= MotionConfig.R_STEP_FAST

    @staticmethod
    def pitch_up(state, t, dt):
        state.tar_omg[1] -= 0.5

    @staticmethod
    def pitch_down(state, t, dt):
        state.tar_omg[1] += 0.5

    @staticmethod
    def dec_vel(state, t, dt):
        if np.linalg.norm(state.vel) < 0.1:
            state.vel[:] = 0
        else:
            state.vel *= (1.0 - 8.6 * dt)

    @staticmethod
    def dec_omg(state, t, dt):
        if np.linalg.norm(state.omg) < 0.1:
            state.omg[:] = 0
        else:
            state.omg *= (1.0 - 9 * dt)