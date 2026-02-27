import numpy as np
from dataclasses import dataclass, field


@dataclass
class MotionConfig:
    T_STEP = 1.5          # 平移加速度步长
    R_STEP_SLOW = 2.0     # 慢速旋转加速度步长（底盘）
    R_STEP_FAST = 8.0     # 快速旋转（小陀螺）
    G_STEP_SLOW = 2.0     # 云台偏航加速度步长
    G_STEP_PITCH = 2.0    # 云台俯仰加速度步长


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
    # === 平移类（作用于底盘） ===
    @staticmethod
    def go_left(state, t, dt):
        state.tar_vel[1] += MotionConfig.T_STEP

    @staticmethod
    def go_right(state, t, dt):
        state.tar_vel[1] -= MotionConfig.T_STEP

    @staticmethod
    def go_up(state, t, dt):
        state.tar_vel[0] += MotionConfig.T_STEP

    @staticmethod
    def go_down(state, t, dt):
        state.tar_vel[0] -= MotionConfig.T_STEP

    @staticmethod
    def ascend(state, t, dt):
        state.tar_vel[2] += MotionConfig.T_STEP

    @staticmethod
    def descend(state, t, dt):
        state.tar_vel[2] -= MotionConfig.T_STEP

    # === 底盘旋转 ===
    @staticmethod
    def rotate_chassis_anticlockwise(state, t, dt):
        state.tar_omg[2] += MotionConfig.R_STEP_SLOW

    @staticmethod
    def rotate_chassis_clockwise(state, t, dt):
        state.tar_omg[2] -= MotionConfig.R_STEP_SLOW

    @staticmethod
    def rotate_chassis_fast_anticlockwise(state, t, dt):
        state.tar_omg[2] += MotionConfig.R_STEP_FAST

    @staticmethod
    def rotate_chassis_fast_clockwise(state, t, dt):
        state.tar_omg[2] -= MotionConfig.R_STEP_FAST

    # === 云台旋转（相对角速度） ===
    # 注意：这些函数应作用于 Gimbal 实体，其状态中应包含 rel_omg 作为 tar_omg 的一部分
    # 我们这里使用 tar_omg 表示目标相对角速度
    @staticmethod
    def rotate_gimbal_yaw_left(state, t, dt):
        state.tar_omg[2] += MotionConfig.G_STEP_SLOW   # 绕 Z 轴（云台 yaw）

    @staticmethod
    def rotate_gimbal_yaw_right(state, t, dt):
        state.tar_omg[2] -= MotionConfig.G_STEP_SLOW

    @staticmethod
    def rotate_gimbal_pitch_up(state, t, dt):
        state.tar_omg[1] -= MotionConfig.G_STEP_PITCH  # 绕 Y 轴（pitch，注意方向）

    @staticmethod
    def rotate_gimbal_pitch_down(state, t, dt):
        state.tar_omg[1] += MotionConfig.G_STEP_PITCH