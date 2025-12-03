import numpy as np
import math

from algorithms.filter.kalman_filter import KalmanFilter
from algorithms.filter.extended_kalman_filter import ExtendedKalmanFilter

from object.model.tjurm.data_and_utils.slide_weight_avg import SlideWeightedAvg
from object.model.tjurm.data_and_utils.math_tools import safe_angle_sub, limit_rad, get_weight_by_theta, get_angle_trans, get_toggle, is_angle_trans


class Antitop:
    def __init__(self):
        self.armor = None
        self.armor_count = 0

        # 三个滤波器
        self._ekf = None  # 主EKF: [x, y, z, v, vz, angle, w, a, theta, omega, beta, r]
        self._kf_center = None  # 中心KF: [x, y, v, angle, w, a]
        self._kf_rotation = None  # 角速度KF: [theta, omega, beta]

        # 参数配置
        self.r_min = 0.15
        self.r_max = 0.4
        self.enable_weighted = False

        # 存储两个装甲板的半径和高度
        self.r = [0.25, 0.25]
        self.z = [0.0, 0.0]
        self.toggle = 0

        # 滑动平均
        self.weighted_z = [SlideWeightedAvg(500), SlideWeightedAvg(500)]

        self.update_count = 0
        self.last_time = None

    def init_model(self, armor, armor_count):
        self.armor = armor
        self.armor_count = armor_count

        # 主EKF状态: [x, y, z, v, vz, angle, w, a, theta, omega, beta, r]
        x0_ekf = np.array([
            armor.world_pos[0],  # x
            armor.world_pos[1],  # y
            armor.world_pos[2],  # z
            0., 0.,  # v, vz
            0., 0., 0.,  # angle, w, a
            armor.world_rpy[2], 0., 0.,  # theta, omega, beta
            0.25  # r
        ])

        # 中心KF状态: [x, y, v, angle, w, a]
        x0_center = np.array([
            armor.world_pos[0],  # x
            armor.world_pos[1],  # y
            0.,  # v
            0.,  # angle
            0., 0.  # w, a
        ])

        # 角速度KF状态: [theta, omega, beta]
        x0_rotation = np.array([
            armor.world_rpy[2],  # theta
            0., 0.  # omega, beta
        ])

        # 初始化协方差矩阵
        P0_ekf = np.diag([0.01, 0.01, 0.01, 0.05, 0.005, 0.05, 0.005, 0.005, 0.02, 0.04, 0.06, 0.001])
        P0_center = np.diag([0.001, 0.001, 0.01, 0.01, 0.1, 0.1])
        P0_rotation = np.diag([1.0, 1.0, 1.0])

        self._ekf = ExtendedKalmanFilter(x0_ekf, P0_ekf)
        self._kf_center = KalmanFilter(x0_center, P0_center)
        self._kf_rotation = KalmanFilter(x0_rotation, P0_rotation)

        # 初始化存储值
        self.z[self.toggle] = armor.world_pos[2]
        self.r[self.toggle] = 0.25
        self.last_time = None
        self.update_count = 0

    def predict(self, dt):
        if dt <= 0:
            dt = 0.01

        # 预测角速度模型
        F_rotation = np.eye(3)
        F_rotation[0, 1] = dt
        F_rotation[1, 2] = dt
        F_rotation[0, 2] = 0.5 * dt * dt
        Q_rotation = np.diag([1.0, 1.0, 1.0])

        self._kf_rotation.predict(F_rotation, Q_rotation)

        # 预测中心模型
        x_center = self._kf_center.x
        F_center = np.eye(6)
        F_center[0, 2] = dt * np.cos(x_center[3])
        F_center[0, 3] = -dt * x_center[2] * np.sin(x_center[3])
        F_center[0, 5] = 0.5 * dt * dt * np.cos(x_center[3])

        F_center[1, 2] = dt * np.sin(x_center[3])
        F_center[1, 3] = dt * x_center[2] * np.cos(x_center[3])
        F_center[1, 5] = 0.5 * dt * dt * np.sin(x_center[3])

        F_center[2, 5] = dt
        F_center[3, 4] = dt

        Q_center = np.diag([0.001, 0.001, 0.01, 0.01, 0.1, 0.1])
        self._kf_center.predict(F_center, Q_center)

        # 预测主EKF
        def f_func(x):
            x_next = np.zeros_like(x)
            x_next[0] = x[0] + dt * x[3] * np.cos(x[5]) + 0.5 * dt * dt * x[7] * np.cos(x[5])
            x_next[1] = x[1] + dt * x[3] * np.sin(x[5]) + 0.5 * dt * dt * x[7] * np.sin(x[5])
            x_next[2] = x[2] + dt * x[4]
            x_next[3] = x[3] + dt * x[7]
            x_next[4] = x[4]
            x_next[5] = x[5] + dt * x[6]
            x_next[6] = x[6]
            x_next[7] = x[7]
            x_next[8] = x[8] + dt * x[9] + 0.5 * dt * dt * x[10]
            x_next[9] = x[9] + dt * x[10]
            x_next[10] = x[10]
            x_next[11] = x[11]
            return x_next

        def F_jacobian(x):
            # 计算f_func的雅可比矩阵
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

        Q_ekf = np.diag([0.01, 0.01, 0.01, 0.05, 0.005, 0.05, 0.005, 0.005, 0.02, 0.04, 0.06, 0.001])
        self._ekf.predict(F=None, Q=Q_ekf, f_func=f_func, F_jacobian=F_jacobian)

        # 同步角速度到主EKF
        self._ekf.x[8] = self._kf_rotation.x[0]
        self._ekf.x[9] = self._kf_rotation.x[1]
        self._ekf.x[10] = self._kf_rotation.x[2]

    def update(self, armor):
        if self._ekf is None:
            return

        current_time = self.last_time if self.last_time is not None else 0
        dt = 0.01  # 默认时间步长

        # 计算theta
        center_yaw = math.atan2(self._kf_center.x[1], self._kf_center.x[0])
        theta = armor.world_rpy[2]

        # 角度转换逻辑
        if self.armor_count == 2:
            self.toggle = 0
        else:
            self.toggle = get_toggle(self.armor_count, self.toggle, theta, self._kf_rotation.x[0])
            if is_angle_trans(self.armor_count, theta, self._kf_rotation.x[0] + self._kf_rotation.x[1] * dt):
                self._kf_rotation.x[0] = theta
                self._ekf.x[8] = theta
                return

        # 更新角速度模型
        predict_theta = (self._kf_rotation.x[0] + self._kf_rotation.x[1] * dt +
                         0.5 * self._kf_rotation.x[2] * dt * dt)
        self._kf_rotation.x[0] = get_angle_trans(self.armor_count, theta, self._kf_rotation.x[0], predict_theta)

        H_rotation = np.array([[1, 0, 0]])
        z_rotation = np.array([theta])
        R_rotation = np.array([[1.0]])

        self._kf_rotation.update(z_rotation, H_rotation, R_rotation)

        # 更新主EKF
        self._ekf.x[2] = self.z[self.toggle]
        self._ekf.x[11] = self.r[self.toggle]

        def h_func(x):
            y = np.zeros(4)
            y[0] = x[0] - x[11] * np.cos(x[8])  # x_armor
            y[1] = x[1] - x[11] * np.sin(x[8])  # y_armor
            y[2] = x[2]  # z
            y[3] = x[8]  # theta
            return y

        def H_jacobian(x):
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

        def z_subtract(z_actual, z_pred):
            result = z_actual - z_pred
            result[3] = safe_angle_sub(z_actual[3], z_pred[3])
            return result

        z_ekf = np.array([
            armor.world_pos[0],
            armor.world_pos[1],
            armor.world_pos[2],
            theta
        ])

        R_ekf = np.diag([0.1, 0.1, 0.1, 0.2])
        self._ekf.update(z=z_ekf, H=None, R=R_ekf, h_func=h_func, H_jacobian=H_jacobian, z_subtract_func=z_subtract)

        # 同步角速度到主EKF
        self._ekf.x[8] = self._kf_rotation.x[0]
        self._ekf.x[9] = self._kf_rotation.x[1]
        self._ekf.x[10] = self._kf_rotation.x[2]

        # 限制半径范围
        self._ekf.x[11] = np.clip(self._ekf.x[11], self.r_min, self.r_max)

        # 更新存储值
        self.z[self.toggle] = self._ekf.x[2]
        self.r[self.toggle] = self._ekf.x[11]

        # 更新中心模型
        H_center = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])
        z_center = np.array([self._ekf.x[0], self._ekf.x[1]])
        R_center = np.diag([1.0, 1.0])

        self._kf_center.update(z_center, H_center, R_center)

        # 角度归一化
        self._kf_center.x[3] = limit_rad(self._kf_center.x[3])
        self._ekf.x[5] = limit_rad(self._ekf.x[5])

        # 更新加权z值
        weight = get_weight_by_theta(theta - center_yaw)
        self.weighted_z[self.toggle].push(armor.world_pos[2], weight)

        self.update_count += 1

        print(self._ekf.x[9])

    def get_est_armor_pos(self, armor_id):
        if self._ekf is None:
            return np.array([0., 0., 0.])

        x = self._ekf.x

        # 计算装甲板角度
        armor_angle = x[8] + armor_id * 2 * math.pi / self.armor_count
        armor_angle = limit_rad(armor_angle)

        # 确定半径和高度偏移
        if self.armor_count == 4 and armor_id % 2 == 1:  # 左右装甲板
            r = self.r[1] if hasattr(self, 'r') and len(self.r) > 1 else x[11]
            z_offs = self.z[1] if hasattr(self, 'z') and len(self.z) > 1 else 0.0
        else:  # 前后装甲板
            r = self.r[0] if hasattr(self, 'r') and len(self.r) > 0 else x[11]
            z_offs = self.z[0] if hasattr(self, 'z') and len(self.z) > 0 else 0.0

        # 计算装甲板位置
        armor_x = x[0] - r * math.cos(armor_angle)
        armor_y = x[1] - r * math.sin(armor_angle)
        armor_z = x[2] + z_offs

        return np.array([armor_x, armor_y, armor_z])

    def get_est_center_pos(self):
        return np.array([self._ekf.x[0], self._ekf.x[1], self._ekf.x[2]])




