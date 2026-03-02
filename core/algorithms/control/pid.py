import numpy as np
from dataclasses import dataclass


@dataclass
class PIDParams:
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0
    max_output: float = np.inf
    max_integral: float = np.inf
    deadband: float = 0.0


class PID:
    """单级PID控制器，支持可变控制周期"""
    def __init__(self, params: PIDParams):
        self.params = params
        self.last_error = 0.0
        self.integral = 0.0
        self.output = 0.0

    def reset(self):
        self.last_error = 0.0
        self.integral = 0.0
        self.output = 0.0

    def update(self, error: float, dt: float) -> float:
        # 死区处理
        if abs(error) < self.params.deadband:
            error = 0.0

        p_term = self.params.kp * error
        self.integral += error * dt
        # 积分限幅
        if self.integral > self.params.max_integral:
            self.integral = self.params.max_integral
        elif self.integral < -self.params.max_integral:
            self.integral = -self.params.max_integral
        i_term = self.params.ki * self.integral
        d_term = self.params.kd * (error - self.last_error) / dt if dt > 0 else 0.0

        output = p_term + i_term + d_term
        # 输出限幅
        if output > self.params.max_output:
            output = self.params.max_output
        elif output < -self.params.max_output:
            output = -self.params.max_output

        self.last_error = error
        self.output = output
        return output


class CascadedPID:
    """串级PID控制器，外环输出作为内环设定值，支持前馈"""
    def __init__(self, outer_params: PIDParams, inner_params: PIDParams):
        self.outer = PID(outer_params)
        self.inner = PID(inner_params)

    def reset(self):
        self.outer.reset()
        self.inner.reset()

    def update(self,
               setpoint: float,
               measurement_outer: float,
               measurement_inner: float,
               feedforward_outer: float = 0.0,
               feedforward_inner: float = 0.0,
               dt: float = 0.01) -> float:
        """
        串级PID更新
        :param setpoint: 外环设定值（期望位置）
        :param measurement_outer: 外环测量值（当前位置）
        :param measurement_inner: 内环测量值（当前速度）
        :param feedforward_outer: 外环前馈（期望速度）
        :param feedforward_inner: 内环前馈（期望加速度）
        :param dt: 控制周期
        :return: 内环输出（期望加速度）
        """
        # 外环误差
        outer_error = setpoint - measurement_outer
        # 外环输出（期望速度） + 外环前馈
        inner_setpoint = self.outer.update(outer_error, dt) + feedforward_outer
        # 内环误差
        inner_error = inner_setpoint - measurement_inner
        # 内环输出（期望加速度） + 内环前馈
        control = self.inner.update(inner_error, dt) + feedforward_inner
        return control


