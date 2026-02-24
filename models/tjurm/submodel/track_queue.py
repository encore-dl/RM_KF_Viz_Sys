import numpy as np
import time
from typing import Optional, Any
from threading import Lock

from core.algorithms.filters.extended_kalman import ExtendedKalmanFilter
from models.tjurm.utils.my_math import (
    limit_rad,
    get_distance
)


# [ x, y, z, v, vz, angle, w, a ]  [ x, y, z ]
# [ 0, 1, 2, 3, 4,    5,   6, 7 ]  [ 0, 1, 2 ]


# ==================== TQstateV4 类 ====================
class TQstateV4:
    """目标状态类"""

    def __init__(self):
        self.last_t = time.time()  # 目标上一次的时间
        self.last_pose = np.zeros(4)  # 目标上一次的位置 [x, y, z, 0]
        self.model = None  # 目标运动模型 (EKF)
        self.count = 0  # 此目标更新计数
        self.keep = 5  # 此目标保持计数
        self.available = False  # 此目标是否可用

    def refresh(self, pose, t):
        """刷新目标状态"""
        self.last_t = t
        self.last_pose = pose.copy()  # last_pose唯一更替的地方
        self.count += 1  # count唯一自增的地方
        self.keep = 20
        self.available = True  # available唯一被true的地方


# ==================== 状态转移和观测函数 ====================
def tq_f_func(x: np.ndarray, dt: float) -> np.ndarray:
    """TrackQueue的状态转移函数"""
    x_next = np.zeros_like(x)
    x_next[0] = x[0] + dt * x[3] * np.cos(x[5]) + 0.5 * dt * dt * x[7] * np.cos(x[5])
    x_next[1] = x[1] + dt * x[3] * np.sin(x[5]) + 0.5 * dt * dt * x[7] * np.sin(x[5])
    x_next[2] = x[2] + dt * x[4]
    x_next[3] = x[3] + dt * x[7]
    x_next[4] = x[4]
    x_next[5] = x[5] + dt * x[6]
    x_next[6] = x[6]
    x_next[7] = x[7]
    return x_next


def tq_F_jacobian(x: np.ndarray, dt: float) -> np.ndarray:
    """TrackQueue状态转移的雅可比矩阵"""
    F = np.eye(8)
    F[0, 3] = dt * np.cos(x[5])
    F[0, 5] = -dt * x[3] * np.sin(x[5]) - 0.5 * dt * dt * x[7] * np.sin(x[5])
    F[0, 7] = 0.5 * dt * dt * np.cos(x[5])

    F[1, 3] = dt * np.sin(x[5])
    F[1, 5] = dt * x[3] * np.cos(x[5]) + 0.5 * dt * dt * x[7] * np.cos(x[5])
    F[1, 7] = 0.5 * dt * dt * np.sin(x[5])

    F[2, 4] = dt
    F[3, 7] = dt
    F[5, 6] = dt
    return F


def tq_h_func(x: np.ndarray) -> np.ndarray:
    """TrackQueue的观测函数"""
    return x[:3]  # 观测x, y, z


def tq_H_jacob(x: np.ndarray) -> np.ndarray:
    """TrackQueue观测的雅可比矩阵"""
    H = np.zeros((3, 8))
    H[0, 0] = 1
    H[1, 1] = 1
    H[2, 2] = 1
    return H


def tq_z_sub(z_actual: np.ndarray, z_pred: np.ndarray) -> np.ndarray:
    """TrackQueue的观测残差计算"""
    return z_actual - z_pred


# ==================== TrackQueueV4 主类 ====================
class TrackQueue:
    """
    多目标跟踪队列
    状态向量: [x, y, z, v, vz, angle, w, a]
    观测向量: [x, y, z]
    """

    def __init__(self, count: int = 10, distance: float = 0.15, delay: float = 0.5,
                 fire_interval: float = 0.05, fire_high_delay: float = 0.02,
                 integrator: Optional[Any] = None):
        """初始化"""
        # 参数配置
        self.count_ = count  # 可维持状态稳定的最小更新次数
        self.distance_ = distance  # 可认为是同一个目标的最大移动距离
        self.delay_ = delay  # 可认为是同一个目标的最大更新延迟
        self.fire_interval = fire_interval  # 射击间隔
        self.fire_high_delay = fire_high_delay  # 开火信号延迟
        self.integrator = integrator  # 自身运动补偿的积分器

        # 状态列表
        self.list_ = []  # 目标状态列表
        self.last_state = None  # 上一次的状态
        self.lock = Lock()  # 线程锁

        # 滤波器参数
        self.matrixQ_ = np.zeros((8, 8))  # 过程噪声协方差矩阵
        self.matrixR_ = np.zeros((3, 3))  # 观测噪声协方差矩阵

        # 初始化协方差矩阵
        self.set_matrix_q(0.1, 0.1, 0.1, 500., 1., 1., 100., 500.)
        self.set_matrix_r(0.1, 0.1, 0.1)

    def set_count(self, count: int):
        """设置认为模型可用的最小更新次数"""
        self.count_ = count

    def set_distance(self, distance: float):
        """设置认为是同一个目标的最大移动距离"""
        self.distance_ = distance

    def set_delay(self, delay: float):
        """设置模型不重置的最大延迟"""
        self.delay_ = delay

    def set_matrix_q(self, q1: float, q2: float, q3: float, q4: float,
                     q5: float, q6: float, q7: float, q8: float):
        """设置过程噪声协方差矩阵"""
        self.matrixQ_ = np.diag([q1, q2, q3, q4, q5, q6, q7, q8])

    def set_matrix_r(self, r1: float, r2: float, r3: float):
        """设置观测噪声协方差矩阵"""
        self.matrixR_ = np.diag([r1, r2, r3])

    def push(self, pose: np.ndarray, t: float):
        """
        推入单次目标信息
        input_pose: [x, y, z, angle]
        t: 时间戳
        """
        with self.lock:
            pose_3d = pose[:3]  # 只取x, y, z

            min_distance = float('inf')
            best_state = None

            # 遍历现有目标，寻找最匹配的
            i = 0
            while i < len(self.list_):
                state = self.list_[i]
                dt = t - state.last_t

                # 检查目标是否过期
                if dt > self.delay_ or state.keep <= 0:
                    if self.last_state == state:
                        self.last_state = None
                    del state
                    self.list_.pop(i)
                    continue

                # 预测目标位置
                if state.model:
                    # 使用状态转移函数预测
                    predict_pose_func = lambda x: tq_f_func(x, dt)[:3]
                    predict_pose = predict_pose_func(state.model.x)

                    # 计算距离
                    distance = get_distance(pose_3d, predict_pose)
                    if distance < min_distance:
                        min_distance = distance
                        best_state = state

                i += 1

            # 如果没有匹配的目标或距离太远，创建新目标
            if best_state is None or min_distance > self.distance_:
                best_state = TQstateV4()
                best_state.last_t = t
                best_state.last_pose = pose.copy()
                best_state.count = 1
                best_state.keep = 20
                best_state.available = True

                # 初始化EKF模型
                x0 = np.array([
                    pose[0],  # x
                    pose[1],  # y
                    pose[2],  # z
                    0.0,  # v
                    0.0,  # vz
                    0.0,  # angle
                    0.0,  # w
                    0.0  # a
                ])

                P0 = np.eye(8) * 0.1  # 初始协方差

                best_state.model = ExtendedKalmanFilter(x0, P0)
                best_state.model.Q = self.matrixQ_
                best_state.model.R = self.matrixR_

                # 初始预测和更新
                dt_pred = 0.0
                F_func = lambda x: tq_F_jacobian(x, dt_pred)
                best_state.model.predict(F_func(x0), self.matrixQ_,
                                         f_func=lambda x: tq_f_func(x, dt_pred),
                                         F_jacobian=F_func)

                best_state.model.update(pose_3d, None, self.matrixR_,
                                        z_sub_func=tq_z_sub,
                                        h_func=tq_h_func,
                                        H_jacob=tq_H_jacob)

                self.list_.append(best_state)
            else:
                # 匹配到现有目标，更新模型
                # 运动补偿
                # integral_x, integral_y = 0.0, 0.0
                # if self.integrator:
                #     integral_x, integral_y = self.integrator.get_integral(
                #         best_state.last_t, t, 0, 0)

                # 补偿底盘运动
                # if best_state.model:
                #     best_state.model.x[0] -= integral_x
                #     best_state.model.x[1] -= integral_y

                # 计算时间间隔
                dt_pred = t - best_state.last_t

                # 刷新目标状态
                best_state.refresh(pose, t)

                # 预测
                if best_state.model:
                    F_func = lambda x: tq_F_jacobian(x, dt_pred)
                    best_state.model.predict(F_func(best_state.model.x), self.matrixQ_,
                                             f_func=lambda x: tq_f_func(x, dt_pred),
                                             F_jacobian=F_func)

                    # 更新
                    best_state.model.update(pose_3d, None, self.matrixR_,
                                            z_sub_func=tq_z_sub,
                                            h_func=tq_h_func,
                                            H_jacob=tq_H_jacob)

                    # 角度归一化
                    best_state.model.x[5] = limit_rad(best_state.model.x[5])

    def update(self):
        """每帧更新一次，减少所有目标的keep值"""
        with self.lock:
            for state in self.list_:
                state.keep -= 1

    def get_pred_armor_pos(self, append_delay: float = 0.0) -> np.ndarray:
        """获取根据模型预测的位姿"""
        with self.lock:
            state = None

            # 首先检查上次使用的状态是否仍然有效
            if self.last_state is not None:
                dt = time.time() - self.last_state.last_t
                if dt < self.delay_ and self.last_state.keep >= 0:
                    state = self.last_state
                else:
                    self.last_state = None

            # 如果没有上次的状态，则寻找count最大的有效状态
            if state is None:
                max_count = -1
                for s in self.list_:
                    dt = time.time() - s.last_t
                    if dt > self.delay_ or s.keep <= 0:
                        continue

                    if s.count > max_count:
                        max_count = s.count
                        state = s

            if state is not None and state.model:
                self.last_state = state  # 记录上次使用的状态

                # 计算总延迟
                sys_delay = time.time() - state.last_t
                total_dt = sys_delay + append_delay

                # 使用状态转移函数预测位置
                x = state.model.x[0] + total_dt * state.model.x[3] * np.cos(state.model.x[5])
                y = state.model.x[1] + total_dt * state.model.x[3] * np.sin(state.model.x[5])
                z = state.model.x[2] + total_dt * state.model.x[4]

                return np.array([x, y, z])  # 角度设为0
            else:
                return np.zeros(3)  # 返回零值

    def get_antitop_input(self) -> tuple:
        """
        获取可击打目标的函数，具备击打目标优先级的排序
        返回: (是否成功, 位姿, 时间)
        """
        with self.lock:
            available_states = []

            # 收集所有可用的状态
            for state in self.list_:
                dt = time.time() - state.last_t
                if dt > self.delay_ or state.keep <= 0:
                    continue

                if state.available:
                    state.available = False  # 标记为已使用
                    if state.count > 2:  # 跟踪达2次以上
                        available_states.append(state)

            # 根据可用状态数量处理
            if len(available_states) == 0:
                return False, np.zeros(4), time.time()
            elif len(available_states) == 1:
                pose = available_states[0].last_pose.copy()
                t = available_states[0].last_t
                return True, pose, t
            else:
                # 按count降序排序，选择更新次数最多的
                available_states.sort(key=lambda s: s.count, reverse=True)
                pose = available_states[0].last_pose.copy()
                t = available_states[0].last_t
                return True, pose, t

