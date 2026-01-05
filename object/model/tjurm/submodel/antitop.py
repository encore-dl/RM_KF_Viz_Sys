import numpy as np
import math
import time
from typing import Optional

from algorithm.kalman_filter.kalman_filter import KalmanFilter
from algorithm.kalman_filter.extended_kalman_filter import ExtendedKalmanFilter

from object.model.tjurm.data_and_utils.slide_integrator import SlideIntegrator
from object.model.tjurm.data_and_utils.slide_weight_avg import SlideWeightedAvg
from object.model.tjurm.data_and_utils.math_tools import (
    get_angle_min,
    safe_angle_sub,
    limit_rad,
    get_weight_by_theta,
    get_angle_trans,
    get_toggle,
    is_angle_trans
)


# main model
# [ x, y, z, v, vz, angle, w, a, theta, omega, beta, r ]  [ x, y, z, theta ]
# [ 0, 1, 2, 3, 4,    5,   6, 7,   8,     9,    10, 11 ]  [ 0, 1, 2,   3   ]

# center model
# [ x, y, v, angle, w, a ]    [ x, y ]
# [ 0, 1, 2,  3,    4, 5 ]    [ 0, 1 ]

# omega model
# [ theta, omega, beta ]    [ theta ]
# [   0,     1,    2   ]    [   0   ]


class Antitop:
    def __init__(self,
                 r_min: float = 0.15,
                 r_max: float = 0.4,
                 armor_count: int = 4,
                 enable_weighted: bool = False,
                 retention_omega: float = 1000.0,
                 fire_interval: float = 0.05,
                 fire_high_delay: float = 0.02,
                 fire_query_resolution: float = 0.002,
                 integrator: Optional[SlideIntegrator] = None
                 ):
        """初始化"""
        # 参数配置
        self.r_min = r_min
        self.r_max = r_max
        self.armor_count = armor_count
        self.enable_weighted = enable_weighted
        self.retention_omega = retention_omega
        self.fire_interval = fire_interval
        self.fire_high_delay = fire_high_delay
        self.fire_query_resolution = fire_query_resolution
        self.integrator = integrator

        # 状态变量
        self.r = [0.25, 0.25]  # 两个位姿的半径
        self.z = [0.2, 0.2]  # 两个位姿的高度
        self.toggle = 0  # 切换标签
        self.update_num = 0  # 更新次数

        # 滤波器
        def main_add(a, b):
            c = a + b
            c[5] = limit_rad(c[5])
            c[8] = limit_rad(c[8])
            return c

        def center_add(a, b):
            c = a + b
            c[3] = limit_rad(c[3])
            return c

        def omega_add(a, b):
            c = a + b
            c[0] = limit_rad(c[0])
            return c

        self.main_model = ExtendedKalmanFilter(
            state_dim=12,
            x_add_func=main_add
        )  # 主EKF模型
        self.center_model = ExtendedKalmanFilter(
            state_dim=6,
            x_add_func=center_add
        )  # 中心KF模型
        self.omega_model = KalmanFilter(
            state_dim=3,
            x_add_func=omega_add
        )  # 角速度KF模型

        # 辅助类
        self.weighted_z = [SlideWeightedAvg(500), SlideWeightedAvg(500)]

        # 时间相关
        self.t = 0.0  # 上一次更新时间
        self.center_last_fire = 0.0  # 中心模式上次开火时间
        self.retention_v_time = 0.0  # 间隔击打速度时间
        self.retention_flag = False  # 间隔击打标志
        self.retention_v_flag = False  # 间隔击打速度标志

        # 开火参数
        self.fire_update = 100
        self.fire_delay = 0.5
        self.fire_armor_angle = 0.5
        self.fire_center_angle = 0.2

        # 初始化协方差矩阵
        self.Q_main = None
        self.Q_center = None
        self.Q_omega = None
        self.R_main = None
        self.R_center = None
        self.R_omega = None

        # self.set_matrix_q(0.01, 0.01, 0.01, 0.1, 0.0001, 0.01,
        #                   0.5, 0.5, 10., 100., 1000., 10000.)
        # self.set_matrix_r(0.1, 0.1, 0.1, 0.2)
        # self.set_center_matrix_q(0.1, 0.1, 1, 1, 10, 10)
        # self.set_center_matrix_r(1.0, 1.0)
        # self.set_omega_matrix_q(10., 100., 1000.)
        # self.set_omega_matrix_r(0.01)

        # self.set_matrix_q(0.1, 0.1, 0.02, 0.5, 0.1, 0.05,
        #                   0.5, 2.0, 0.02, 1.0, 5.0, 0.001)
        # self.set_matrix_r(0.001, 0.001, 0.001, 0.0001)
        # self.set_center_matrix_q(0.1, 0.1, 0.3, 0.03, 0.3, 1.0)
        # self.set_center_matrix_r(0.002, 0.002)
        # self.set_omega_matrix_q(0.005, 0.3, 1.0)
        # self.set_omega_matrix_r(0.00005)

        self.set_matrix_q(0.01, 0.01, 0.01, 0.05, 0.005, 0.05,
                          0.005, 0.005, 0.02, 0.04, 0.06, 0.001)
        self.set_matrix_r(1, 1, 0.01, 0.02)
        self.set_center_matrix_q(0.001, 0.001, 0.01, 0.01, 0.1, 0.1)
        self.set_center_matrix_r(1.0, 1.0)
        self.set_omega_matrix_q(1.0, 1.0, 1.0)
        self.set_omega_matrix_r(1.0)

        # main model
        # [ x, y, z, v, vz, angle, w, a, theta, omega, beta, r ]  [ x, y, z, theta ]
        # [ 0, 1, 2, 3, 4,    5,   6, 7,   8,     9,    10, 11 ]  [ 0, 1, 2,   3   ]
        # center model
        # [ x, y, v, angle, w, a ]    [ x, y ]
        # [ 0, 1, 2,  3,    4, 5 ]    [ 0, 1 ]
        # omega model
        # [ theta, omega, beta ]    [ theta ]
        # [   0,     1,    2   ]    [   0   ]

    def set_matrix_q(self, q0: float, q1: float, q2: float, q3: float,
                     q4: float, q5: float, q6: float, q7: float,
                     q8: float, q9: float, q10: float, q11: float):
        """设置主模型过程噪声协方差矩阵"""
        self.Q_main = np.diag([q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11])

    def set_matrix_r(self, r0: float, r1: float, r2: float, r3: float):
        """设置主模型观测噪声协方差矩阵"""
        self.R_main = np.diag([r0, r1, r2, r3])

    def set_center_matrix_q(self, q0: float, q1: float, q2: float,
                            q3: float, q4: float, q5: float):
        """设置中心模型过程噪声协方差矩阵"""
        self.Q_center = np.diag([q0, q1, q2, q3, q4, q5])

    def set_center_matrix_r(self, r0: float, r1: float):
        """设置中心模型观测噪声协方差矩阵"""
        self.R_center = np.diag([r0, r1])

    def set_omega_matrix_q(self, q0: float, q1: float, q2: float):
        """设置角速度模型过程噪声协方差矩阵"""
        self.Q_omega = np.diag([q0, q1, q2])

    def set_omega_matrix_r(self, r0: float):
        """设置角速度模型观测噪声协方差矩阵"""
        self.R_omega = np.array([[r0]])

    def push(self, pose, t: float):
        """
        推送新观测值
        pose: [x, y, z, theta]
        """
        # 底盘运动补偿
        # integral_x, integral_y = 0.0, 0.0
        # if self.integrator:
        #     integral_x, integral_y = self.integrator.get_integral(self.t, t)

        # 自适应角度融合
        x, y, z, theta_obs = pose[0], pose[1], pose[2], pose[3]

        # 系数计算：1-0.2ω or 0.2
        if abs(self.omega_model.x[1]) < 4.0:
            coeff = 1 - abs(self.omega_model.x[1]) / 5.0
        else:
            coeff = 0.2

        # 几何计算的角度
        if self.main_model is not None:
            theta_geo = math.atan2(self.main_model.x[1] - y, self.main_model.x[0] - x)
        else:
            theta_geo = theta_obs

        # 自适应融合角度
        theta = coeff * theta_obs + (1 - coeff) * theta_geo

        # 计算时间间隔 检查延迟
        dt = t - self.t
        if dt <= 0:
            dt = 0.001
        if dt > self.fire_delay:
            self.update_num = 0
            self.main_model.restart()
        self.update_num += 1
        self.t = t

        # 处理装甲板切换
        if self.armor_count == 2:
            self.toggle = 0
            if dt > 0.05:
                self.omega_model.x[0] = theta
                if self.main_model:
                    self.main_model.x[8] = theta
                return
        else:
            self.toggle = get_toggle(self.armor_count, self.toggle, theta, self.omega_model.x[0])

            # 检查角度转换临界状态
            if is_angle_trans(self.armor_count, theta, self.omega_model.x[0] + self.omega_model.x[1] * dt + 0.5 * self.omega_model.x[2] * dt * dt):
                self.omega_model.x[0] = theta
                if self.main_model:
                    self.main_model.x[8] = theta
                return

        # ========== 角速度模型预测和更新 ==========
        # 角度转换
        self.omega_model.x[0] = get_angle_trans(
            self.armor_count,
            theta,
            self.omega_model.x[0],
            self.omega_model.x[0] + self.omega_model.x[1] * dt + 0.5 * self.omega_model.x[2] * dt * dt
        )

        # 预测 - 使用线性预测（与C++一致）
        F_omega = np.eye(3)
        F_omega[0, 1] = dt
        F_omega[1, 2] = dt
        F_omega[0, 2] = 0.5 * dt * dt
        self.omega_model.predict(F=F_omega, Q=self.Q_omega)

        # 更新
        H_omega = np.array([[1, 0, 0]])
        z_omega = np.array([theta])

        def z_sub_omega(z_actual, z_pred):
            result = z_actual - z_pred
            result[0] = safe_angle_sub(z_actual[0], z_pred[0])
            return result

        self.omega_model.update(z_omega, H_omega, self.R_omega, z_sub_omega)

        # ========== 主模型预测和更新 ==========
        if self.main_model:
            # 补偿底盘运动
            # self.main_model.x[0] -= integral_x
            # self.main_model.x[1] -= integral_y

            # 角度转换
            self.main_model.x[8] = get_angle_trans(
                self.armor_count,
                theta,
                self.main_model.x[8],
                self.main_model.x[8] + self.main_model.x[9] * dt + 0.5 * self.main_model.x[10] * dt * dt
            )

            # 使用记忆的参数
            self.main_model.x[2] = self.z[self.toggle]
            self.main_model.x[11] = self.r[self.toggle]

            # 预测
            def f_func_main(x):
                x_next = np.zeros_like(x)
                x_next[0] = x[0] + dt * x[3] * np.cos(x[5]) + 0.5 * dt * dt * x[7] * np.cos(x[5])
                x_next[1] = x[1] + dt * x[3] * np.sin(x[5]) + 0.5 * dt * dt * x[7] * np.sin(x[5])
                x_next[2] = x[2] + dt * x[4]
                x_next[3] = x[3] + dt * x[7]
                x_next[4] = x[4]
                x_next[5] = x[5] + dt * x[6]
                x_next[6] = x[6]
                x_next[7] = x[7] * 0.9
                x_next[8] = x[8] + dt * x[9] + 0.5 * dt * dt * x[10]
                x_next[9] = x[9] + dt * x[10]
                x_next[10] = x[10] * 0.9
                x_next[11] = x[11]
                return x_next

            def F_jacobian_main(x):
                F = np.eye(12)
                F[0, 3] = dt * np.cos(x[5])
                F[0, 5] = -dt * x[3] * np.sin(x[5]) - 0.5 * dt * dt * x[7] * np.sin(x[5])
                F[0, 7] = 0.5 * dt * dt * np.cos(x[5])

                F[1, 3] = dt * np.sin(x[5])
                F[1, 5] = dt * x[3] * np.cos(x[5]) + 0.5 * dt * dt * x[7] * np.cos(x[5])
                F[1, 7] = 0.5 * dt * dt * np.sin(x[5])

                F[2, 4] = dt
                F[3, 7] = dt
                F[5, 6] = dt
                F[8, 9] = dt
                F[8, 10] = 0.5 * dt * dt
                F[9, 10] = dt
                return F

            self.main_model.predict(F=None, Q=self.Q_main,
                               f_func=f_func_main, F_jacobian=F_jacobian_main)

            # 更新
            def h_func_main(x):
                y = np.zeros(4)
                y[0] = x[0] - x[11] * np.cos(x[8])  # x_armor
                y[1] = x[1] - x[11] * np.sin(x[8])  # y_armor
                y[2] = x[2]  # z
                y[3] = x[8]  # theta
                return y

            def H_jacob_main(x):
                H = np.zeros((4, 12))
                H[0, 0] = 1  # dx_armor/dx
                H[0, 8] = x[11] * np.sin(x[8])  # dx_armor/dtheta
                H[0, 11] = -np.cos(x[8])  # dx_armor/dr

                H[1, 1] = 1  # dy_armor/dy
                H[1, 8] = -x[11] * np.cos(x[8])  # dy_armor/dtheta
                H[1, 11] = -np.sin(x[8])  # dy_armor/dr

                H[2, 2] = 1  # dz/dz
                H[3, 8] = 1  # dtheta/dtheta
                return H

            def z_sub_main(z_actual, z_pred):
                result = z_actual - z_pred
                result[3] = safe_angle_sub(z_actual[3], z_pred[3])
                return result

            z_main = np.array([x, y, z, theta])
            self.main_model.update(
                z_main,
                H=None,
                R=self.R_main,
                h_func=h_func_main,
                H_jacob=H_jacob_main,
                z_sub_func=z_sub_main
            )

            # 同步角速度
            self.main_model.x[8] = self.omega_model.x[0]
            self.main_model.x[9] = self.omega_model.x[1]
            self.main_model.x[10] = self.omega_model.x[2]

            # 限制半径
            self.main_model.x[11] = np.clip(self.main_model.x[11], self.r_min, self.r_max)

            # 更新记忆参数
            self.z[self.toggle] = self.main_model.x[2]
            self.r[self.toggle] = self.main_model.x[11]

        # ========== 中心模型预测和更新 ==========
        if self.center_model and self.main_model:
            # 底盘运动补偿
            # self.center_model.x[0] -= integral_x
            # self.center_model.x[1] -= integral_y

            # 预测 - 使用非线性函数
            def f_func_center(x):
                x_next = np.zeros_like(x)
                x_next[0] = x[0] + dt * x[2] * np.cos(x[3]) + 0.5 * dt * dt * x[5] * np.cos(x[3])
                x_next[1] = x[1] + dt * x[2] * np.sin(x[3]) + 0.5 * dt * dt * x[5] * np.sin(x[3])
                x_next[2] = x[2] + dt * x[5]
                x_next[3] = x[3] + dt * x[4]
                x_next[4] = x[4]
                x_next[5] = x[5]
                return x_next

            def F_jacobian_center(x):
                F = np.eye(6)
                F[0, 2] = dt * np.cos(x[3])
                F[0, 3] = -dt * x[2] * np.sin(x[3]) - 0.5 * dt * dt * x[5] * np.sin(x[3])
                F[0, 5] = 0.5 * dt * dt * np.cos(x[3])

                F[1, 2] = dt * np.sin(x[3])
                F[1, 3] = dt * x[2] * np.cos(x[3]) + 0.5 * dt * dt * x[5] * np.cos(x[3])
                F[1, 5] = 0.5 * dt * dt * np.sin(x[3])

                F[2, 5] = dt
                F[3, 4] = dt
                return F

            self.center_model.predict(
                F=None,
                Q=self.Q_center,
                f_func=f_func_center,
                F_jacobian=F_jacobian_center
            )

            # 更新 - 线性观测
            H_center = np.array([[1, 0, 0, 0, 0, 0],
                                 [0, 1, 0, 0, 0, 0]])
            z_center = np.array([self.main_model.x[0], self.main_model.x[1]])

            self.center_model.update(
                z_center,
                H=H_center,
                R=self.R_center,
                h_func=None,
                H_jacob=None,
                z_sub_func=lambda a, b: a - b
            )

            # 角度归一化
            self.center_model.x[3] = limit_rad(self.center_model.x[3])
            self.main_model.x[5] = limit_rad(self.main_model.x[5])

        # 更新加权z值
        if self.center_model:
            center_yaw = math.atan2(self.center_model.x[1], self.center_model.x[0])
            weight = get_weight_by_theta(pose[3] - center_yaw)
            self.weighted_z[self.toggle].push(pose[2], weight)

    def get_pred_locking_mode(self, append_delay: float = 0.0):
        """获取预测位姿"""
        if self.main_model is None or self.center_model is None or self.omega_model is None:
            return np.zeros(3)  # 预判断

        current_time = time.time()
        sys_delay = current_time - self.t

        if sys_delay > self.fire_delay:
            return np.zeros(3)

        dt = sys_delay + append_delay

        # 中心位置预测（匀加速模型）
        x_center = (self.center_model.x[0] +
                    self.center_model.x[2] * np.cos(self.center_model.x[3]) * dt +
                    0.5 * self.center_model.x[5] * np.cos(self.center_model.x[3]) * dt * dt)

        y_center = (self.center_model.x[1] +
                    self.center_model.x[2] * np.sin(self.center_model.x[3]) * dt +
                    0.5 * self.center_model.x[5] * np.sin(self.center_model.x[3]) * dt * dt)

        kf_theta = (self.omega_model.x[0] +
                    self.omega_model.x[1] * dt +
                    0.5 * self.omega_model.x[2] * dt * dt)

        # 获取最小角度
        theta = get_angle_min(kf_theta, x_center, y_center, self.armor_count)

        # 角度修正
        omega = self.omega_model.x[1]
        target_yaw = math.atan2(y_center, x_center)

        if omega > 0:
            if safe_angle_sub(theta, target_yaw) > math.pi / 8:
                theta = safe_angle_sub(target_yaw, math.pi / 4)
        else:
            if safe_angle_sub(theta, target_yaw) < -math.pi / 8:
                theta = safe_angle_sub(target_yaw, -math.pi / 4)

        # 获取半径和高度
        toggle_idx = get_toggle(self.armor_count, self.toggle, theta, kf_theta)
        r = self.r[toggle_idx]

        if self.enable_weighted and self.weighted_z[toggle_idx].get_size() >= 20:
            z = self.weighted_z[toggle_idx].get_avg()
        else:
            z = self.z[toggle_idx]

        # 计算装甲板位置
        x = x_center - r * np.cos(theta)
        y = y_center - r * np.sin(theta)

        return [x, y, z]

    def get_pred_waiting_mode(self, append_delay: float = 0.0):
        """获取中心位姿"""
        if self.main_model is None or self.center_model is None or self.omega_model is None:
            return np.zeros(3)

        current_time = time.time()
        sys_delay = current_time - self.t

        if sys_delay > self.fire_delay:
            return np.zeros(3)

        dt = sys_delay + append_delay

        # 中心位置预测
        x_center = (self.center_model.x[0] +
                    self.center_model.x[2] * np.cos(self.center_model.x[3]) * dt +
                    0.5 * self.center_model.x[5] * np.cos(self.center_model.x[3]) * dt * dt)

        y_center = (self.center_model.x[1] +
                    self.center_model.x[2] * np.sin(self.center_model.x[3]) * dt +
                    0.5 * self.center_model.x[5] * np.sin(self.center_model.x[3]) * dt * dt)

        kf_theta = (self.omega_model.x[0] +
                    self.omega_model.x[1] * dt +
                    0.5 * self.omega_model.x[2] * dt * dt)

        theta = get_angle_min(kf_theta, x_center, y_center, self.armor_count)

        # 获取高度
        toggle_idx = get_toggle(self.armor_count, self.toggle, theta, kf_theta)
        if self.enable_weighted and self.weighted_z[toggle_idx].get_size() >= 20:
            z = self.weighted_z[toggle_idx].get_avg()
        else:
            z = self.z[toggle_idx]

        r = self.r[toggle_idx]

        # 计算目标点
        target_yaw = math.atan2(y_center, x_center)
        x = x_center - r * np.cos(target_yaw)
        y = y_center - r * np.sin(target_yaw)

        return [x, y, z]

    def get_pred_robot_pos(self, append_delay):
        if self.main_model is None or self.center_model is None or self.omega_model is None:
            return np.zeros(3)

        current_time = time.time()
        sys_delay = current_time - self.t

        if sys_delay > self.fire_delay:
            return np.zeros(3)

        dt = sys_delay + append_delay

        # 中心位置预测
        x_center = (self.center_model.x[0] +
                    self.center_model.x[2] * np.cos(self.center_model.x[3]) * dt +
                    0.5 * self.center_model.x[5] * np.cos(self.center_model.x[3]) * dt * dt)

        y_center = (self.center_model.x[1] +
                    self.center_model.x[2] * np.sin(self.center_model.x[3]) * dt +
                    0.5 * self.center_model.x[5] * np.sin(self.center_model.x[3]) * dt * dt)

        return [x_center, y_center, sum(self.z)/len(self.z)]

    def get_omega(self):
        if self.omega_model is not None:
            return self.omega_model.x[1]