# import numpy as np
# from typing import Tuple
#
# from object.model.tjurm.data_and_utils.config import TJURMConfig
# from object.model.tjurm.submodel.track_queue import TrackQueueV4
# from object.model.tjurm.submodel.antitop import AntitopV
#
#
# class TJURMModel:
#     """TJURM主模型，整合TrackQueue和Antitop"""
#
#     def __init__(self):
#         self.config = TJURMConfig()
#         self.track_queue = TrackQueueV4(self.config)
#         self.antitop = AntitopV4(self.config)
#
#         self.flag_antitop = False
#         self.flag_center = False
#
#         self.last_armor_id = None
#         self.armor_count = 0
#         self.update_count = 0
#
#     def init_model(self, armor, armor_count: int):
#         """初始化模型"""
#         self.armor_count = armor_count
#         self.last_armor_id = armor.armor_id
#
#         # 初始化状态
#         init_pose = np.array([
#             armor.world_pos[0],
#             armor.world_pos[1],
#             armor.world_pos[2],
#             armor.world_rpy[2]  # yaw
#         ])
#
#         self.track_queue.push(init_pose, 0.0)
#         self.antitop.push(init_pose, 0.0)
#
#     def predict(self, dt: float):
#         """预测步骤"""
#         # TrackQueue和Antitop在push时进行预测，这里预留接口
#         pass
#
#     def update(self, armor):
#         """更新步骤"""
#         pose = np.array([
#             armor.world_pos[0],
#             armor.world_pos[1],
#             armor.world_pos[2],
#             armor.world_rpy[2]
#         ])
#
#         current_time = self.update_count * 0.01  # 简化时间戳
#
#         self.track_queue.push(pose, current_time)
#         self.antitop.push(pose, current_time)
#
#         self.last_armor_id = armor.armor_id
#         self.update_count += 1
#
#     def get_target_pose(self, fly_delay: float, rotate_delay: float,
#                         shoot_delay: float) -> Tuple[np.ndarray, np.ndarray, bool]:
#         """获取目标位姿和射击模式"""
#         # 获取角速度
#         omega = self.antitop.get_omega()
#
#         # 模式切换逻辑
#         if abs(omega) > self.config.track_to_antitop:
#             self.flag_antitop = True
#         elif abs(omega) < self.config.antitop_to_track:
#             self.flag_antitop = False
#
#         if abs(omega) > self.config.armor_to_center:
#             self.flag_center = True
#         elif abs(omega) < self.config.center_to_armor:
#             self.flag_center = False
#
#         # 获取旋转和射击位姿
#         pose_rotate = self.track_queue.get_pose(fly_delay + rotate_delay)
#         pose_shoot = self.track_queue.get_pose(fly_delay + shoot_delay)
#         is_single_shot = False
#
#         if self.flag_antitop:
#             if self.flag_center:
#                 # 中心模式
#                 pose_shoot = self.antitop.get_center(fly_delay + shoot_delay)
#                 pose_rotate = self.antitop.get_center(fly_delay + rotate_delay)
#                 is_single_shot = True
#                 can_fire = self.antitop.get_fire_center(pose_shoot)
#             else:
#                 # 装甲板模式
#                 pose_shoot = self.antitop.get_pose(fly_delay + shoot_delay)
#                 pose_rotate = self.antitop.get_pose(fly_delay + rotate_delay)
#                 is_single_shot = False
#                 can_fire = self.antitop.get_fire_armor(pose_shoot)
#         else:
#             # 跟踪模式
#             is_single_shot = False
#             can_fire = True
#
#         return pose_rotate, pose_shoot, is_single_shot
#
#     def get_est_armor_pos(self, x: np.ndarray, armor_id: int) -> np.ndarray:
#         """获取估计的装甲板位置"""
#         # 这里实现装甲板位置估计逻辑
#         # 简化实现，返回中心位置
#         return np.array([x[0], x[1], x[2]])
#
#     def diverged(self) -> bool:
#         """检查是否发散"""
#         # 简化的发散检查
#         return False
#
#     def converged(self) -> bool:
#         """检查是否收敛"""
#         return self.update_count > 10
#
#     def get_ekf(self):
#         """获取EKF实例"""
#         return self.antitop.ekf