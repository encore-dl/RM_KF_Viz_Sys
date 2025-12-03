# import numpy as np
#
# from algorithms.filter.extended_kalman_filter import ExtendedKalmanFilter
# from object.model.tjurm.data_and_utils.config import TJURMConfig
#
#
# class TrackQueueV4:
#     """TrackQueue V4 Python实现"""
#
#     def __init__(self, config: TJURMConfig):
#         self.config = config
#
#         # 状态向量: [x, y, z, v, vz, angle, w, a]
#         self.state_dim = 8
#         self.obs_dim = 3
#
#         # 初始化EKF
#         self.ekf = ExtendedKalmanFilter()
#         self._init_matrices()
#
#         self.last_pose = None
#         self.last_time = None
#         self.update_count = 0
#         self.available = False
#
#     def _init_matrices(self):
#         """初始化过程噪声和观测噪声矩阵"""
#         # 过程噪声协方差矩阵 Q
#         self.ekf.Q = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
#
#         # 观测噪声协方差矩阵 R
#         self.ekf.R = np.diag([0.1, 0.1, 0.1])
#
#     def _motion_model(self, x: np.ndarray, dt: float) -> np.ndarray:
#         """运动模型: TrackQueueV4_FuncA"""
#         x_next = np.zeros_like(x)
#
#         x_next[0] = x[0] + dt * x[3] * np.cos(x[5]) + 0.5 * dt * dt * x[7] * np.cos(x[5])
#         x_next[1] = x[1] + dt * x[3] * np.sin(x[5]) + 0.5 * dt * dt * x[7] * np.sin(x[5])
#         x_next[2] = x[2] + dt * x[4]
#         x_next[3] = x[3] + dt * x[7]
#         x_next[4] = x[4]
#         x_next[5] = x[5] + dt * x[6]
#         x_next[6] = x[6]
#         x_next[7] = x[7]
#
#         return x_next
#
#     def _observation_model(self, x: np.ndarray) -> np.ndarray:
#         """观测模型: TrackQueueV4_FuncH"""
#         return x[:3]  # 观测位置 [x, y, z]
#
#     def push(self, pose: np.ndarray, timestamp: float):
#         """推入观测数据"""
#         if self.last_time is None:
#             dt = 0.01
#         else:
#             dt = timestamp - self.last_time
#
#         # 初始化状态
#         if self.last_pose is None:
#             self.ekf.x = np.array([
#                 pose[0], pose[1], pose[2],  # x, y, z
#                 0, 0,  # v, vz
#                 pose[3], 0, 0  # angle, w, a
#             ])
#
#         # 预测步骤
#         def f_func(x):
#             return self._motion_model(x, dt)
#
#         def F_jacobian(x):
#             F = np.eye(8)
#             F[0, 3] = dt * np.cos(x[5])
#             F[0, 5] = -dt * x[3] * np.sin(x[5]) - 0.5 * dt * dt * x[7] * np.sin(x[5])
#             F[0, 7] = 0.5 * dt * dt * np.cos(x[5])
#
#             F[1, 3] = dt * np.sin(x[5])
#             F[1, 5] = dt * x[3] * np.cos(x[5]) + 0.5 * dt * dt * x[7] * np.cos(x[5])
#             F[1, 7] = 0.5 * dt * dt * np.sin(x[5])
#
#             F[2, 4] = dt
#
#             F[3, 7] = dt
#             F[5, 6] = dt
#
#             return F
#
#         self.ekf.predict(F=None, Q=self.ekf.Q, f_func=f_func, F_jacobian=F_jacobian)
#
#         # 更新步骤
#         z = pose[:3]  # 观测位置
#
#         def h_func(x):
#             return self._observation_model(x)
#
#         def z_subtract(z_actual, z_pred):
#             return z_actual - z_pred
#
#         H = np.zeros((3, 8))
#         H[0, 0] = 1  # dx/dx
#         H[1, 1] = 1  # dy/dy
#         H[2, 2] = 1  # dz/dz
#
#         self.ekf.update(
#             z=z,
#             H=H,
#             R=self.ekf.R,
#             h_func=h_func,
#             z_subtract_func=z_subtract
#         )
#
#         # 更新状态
#         self.last_pose = pose
#         self.last_time = timestamp
#         self.update_count += 1
#         self.available = True
#
#     def get_pose(self, append_delay: float = 0.0) -> np.ndarray:
#         """获取预测位姿"""
#         if not self.available or self.last_time is None:
#             return np.zeros(4)
#
#         current_time = self.last_time  # 简化处理
#         dt = append_delay
#
#         x = self.ekf.x
#
#         # 预测位置
#         pred_x = x[0] + dt * x[3] * np.cos(x[5])
#         pred_y = x[1] + dt * x[3] * np.sin(x[5])
#         pred_z = x[2] + dt * x[4]
#
#         return np.array([pred_x, pred_y, pred_z, x[5]])

