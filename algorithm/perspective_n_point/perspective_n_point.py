import cv2
import numpy as np
import math
from dataclasses import dataclass

from utils.math_tool import rotation_matrix_to_euler, euler_to_rotation_matrix


# ==================== 配置类 ====================
@dataclass
class PnPConfig:
    """装甲板尺寸配置（单位：米）"""
    # 大装甲板尺寸（230mm x 55mm）
    width_large: float = 0.230  # 灯条间距
    height_large: float = 0.055  # 灯条长度

    # 小装甲板尺寸（135mm x 55mm）
    width_small: float = 0.135  # 灯条间距
    height_small: float = 0.055  # 灯条长度


# ==================== 自定义YawPnP求解器 ====================
class YawPnP:
    """
    1D-PnP 求解器的 Python 实现
    通过优化偏航角(yaw)来改进PnP求解精度
    """

    def __init__(self, intrinsic_matrix):
        # 相机内参矩阵
        self.Kc = intrinsic_matrix

        # 存储数据
        self.P_world = []  # 世界坐标系下的3D点
        self.P_pixel = []  # 图像坐标系下的2D点

        # 姿态参数
        self.sys_yaw = 0.0  # 系统初始偏航角
        self.pose = np.zeros(4)  # 位置向量
        self.T = np.eye(4)  # 变换矩阵
        self.T_inv = np.eye(4)  # 逆变换矩阵
        self.elevation = 0  # 仰角模式

        # 仰角阈值（弧度）
        self.ANGLE_UP_15 = 15.0 * math.pi / 180.0
        self.ANGLE_UP_75 = 75.0 * math.pi / 180.0
        self.ANGLE_DOWN_15 = -15.0 * math.pi / 180.0

    def set_world_points(self, object_points):
        """设置世界坐标系下的3D点（FLU坐标系 -> 内部RDF坐标系）"""
        self.P_world = []
        for p in object_points:
            # FLU坐标系: [0, y左+, z上+]
            # 转换为RDF坐标系: [0, -y右+, -z下+]
            width_val = p[1]  # y方向（宽）
            height_val = p[2]  # z方向（高）
            self.P_world.append(np.array([0, -width_val, -height_val, 1.0]))

    def set_image_points(self, image_points):
        """设置图像坐标系下的2D点"""
        self.P_pixel = [p for p in image_points]

    def set_elevation(self, pitch_rad):
        """根据俯仰角设置仰角模式"""
        if pitch_rad > 0.5:  # 大仰角
            self.elevation = 1
        elif pitch_rad < -0.2:  # 大俯角
            self.elevation = 2
        else:  # 中等角度
            self.elevation = 0

    def get_mapping(self, append_yaw):
        """根据偏航角计算3D点的映射位置"""
        yaw = self.sys_yaw + append_yaw

        # 根据仰角模式选择俯仰角
        if self.elevation == 0:
            pitch = self.ANGLE_UP_15
        elif self.elevation == 1:
            pitch = self.ANGLE_UP_75
        else:
            pitch = self.ANGLE_DOWN_15
        pitch = -pitch  # 符号调整

        # 计算三角函数
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)

        # 构建变换矩阵（局部装甲板坐标系 -> 世界坐标系）
        M = np.array([
            [cy * cp, -sy, -sp * cy, self.pose[0]],
            [sy * cp, cy, -sp * sy, self.pose[1]],
            [sp, 0, cp, self.pose[2]],
            [0, 0, 0, 1]
        ])

        return [M @ p for p in self.P_world]

    def get_project(self, P_mapping):
        """将3D点投影到图像平面"""
        P_project = []
        for p in P_mapping:
            p_camera = (self.T_inv @ p)[:3]  # 转换到相机坐标系
            p_proj_3d = self.Kc @ p_camera  # 投影到图像平面
            if p_camera[2] > 1e-3:  # 避免除零
                P_project.append(p_proj_3d[:2] / p_camera[2])
            else:
                P_project.append(np.array([0.0, 0.0]))
        return P_project

    def get_cost(self, append_yaw, mode='pixel'):
        """计算代价函数"""
        P_map = self.get_mapping(append_yaw)  # 获取映射点
        P_proj = self.get_project(P_map)  # 获取投影点

        cost = 0.0
        map_idx = [0, 1, 3, 2]  # 对应的点序（左上、右上、左下、右下）

        for i in range(4):
            idx_this = map_idx[i]
            idx_next = map_idx[(i + 1) % 4]

            pix_diff = self.P_pixel[idx_next] - self.P_pixel[idx_this]
            proj_diff = P_proj[idx_next] - P_proj[idx_this]

            if mode == 'pixel':
                # 像素距离模式
                this_dist = np.linalg.norm(self.P_pixel[idx_this] - P_proj[idx_this])
                next_dist = np.linalg.norm(self.P_pixel[idx_next] - P_proj[idx_next])
                line_len_diff = abs(np.linalg.norm(pix_diff) - np.linalg.norm(proj_diff))
                base_len = np.linalg.norm(pix_diff) + 1e-6
                cost += (0.5 * (this_dist + next_dist) + line_len_diff) / base_len
            else:
                # 角度模式
                norm_p = np.linalg.norm(pix_diff)
                norm_proj = np.linalg.norm(proj_diff)
                if norm_p * norm_proj > 1e-6:
                    cos_a = np.dot(pix_diff, proj_diff) / (norm_p * norm_proj)
                    cost += abs(math.acos(max(-1.0, min(1.0, cos_a))))
        return cost

    def ternary_search(self, mode, left, right, epsilon=0.03):
        """三分搜索法寻找最优偏航角"""
        while (right - left) > epsilon:
            m1 = left + (right - left) / 3
            m2 = right - (right - left) / 3
            if self.get_cost(m1, mode) < self.get_cost(m2, mode):
                right = m2
            else:
                left = m1
        return (left + right) / 2


# ==================== 主求解函数 ====================
def solve_pnp_core(camera_k, camera_dist, armor_type, image_points_2d,
                   head_yaw, trans_head2world, rot_head2world,
                   trans_pnp2head, rot_pnp2head, use_plus_pnp=True):
    """
    PnP求解核心函数

    参数:
        camera_k: 相机内参矩阵
        camera_dist: 相机畸变系数
        armor_type: 装甲板类型 ('large' 或 'small')
        image_points_2d: 图像上的4个角点坐标
        head_yaw: 云台当前偏航角
        trans_head2world: 云台到世界坐标系的平移
        rot_head2world: 云台到世界坐标系的旋转
        trans_pnp2head: PnP坐标系到云台的平移
        rot_pnp2head: PnP坐标系到云台的旋转
        use_plus_pnp: 是否使用增强PnP

    返回:
        final_pos: 装甲板在世界坐标系中的位置
        final_yaw: 装甲板的偏航角
    """

    # 1. 初始化配置和装甲板模型点
    cfg = PnPConfig()
    w, h = (cfg.width_large, cfg.height_large) if armor_type == 'large' else (cfg.width_small, cfg.height_small)

    # 3D模型点（装甲板中心为原点，FLU坐标系）
    obj_pts = np.array([
        [0, w / 2, h / 2],  # 左上
        [0, -w / 2, h / 2],  # 右上
        [0, -w / 2, -h / 2],  # 右下
        [0, w / 2, -h / 2]  # 左下
    ])

    img_pts = np.ascontiguousarray(image_points_2d).astype(np.float64)

    # 2. 标准OpenCV PnP求解
    success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, camera_k, camera_dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success or np.isnan(rvec).any() or np.isnan(tvec).any():
        print(not success, np.isnan(rvec).any(), np.isnan(tvec).any())
        return None, None

    # 旋转向量转换为旋转矩阵
    rot_pnp, _ = cv2.Rodrigues(rvec)

    # 3. 构建变换矩阵链
    T_pnp = np.eye(4)
    T_pnp[:3, :3] = rot_pnp
    T_pnp[:3, 3] = tvec.flatten()

    T_p2h = np.eye(4)
    T_p2h[:3, :3] = rot_pnp2head
    T_p2h[:3, 3] = trans_pnp2head

    T_h2w = np.eye(4)
    T_h2w[:3, :3] = rot_head2world
    T_h2w[:3, 3] = trans_head2world

    # 4. 计算最终位姿
    T_world = T_h2w @ T_p2h @ T_pnp
    final_pos = T_world[:3, 3]

    if np.isnan(final_pos).any():
        print("计算结果包含NaN")
        return None, None

    # 5. 计算偏航角
    rot_world = T_world[:3, :3]
    # vec_z = rot_world @ np.array([0, 0, 1])
    # final_yaw = math.atan2(vec_z[1], vec_z[0])
    final_rpy = rotation_matrix_to_euler(rot_world)

    # 如果不需要增强PnP，直接返回标准结果
    if not use_plus_pnp:
        return final_pos, final_rpy

    # # 增强PnP：在标准PnP的基础上微调yaw
    # # 方法：使用标准PnP的roll和pitch，只优化yaw
    #
    # # 1. 从标准PnP提取yaw作为基准
    # base_yaw = final_rpy[2]  # 这是装甲板在世界坐标系中的yaw
    #
    # # 2. 构建一个简单的优化函数
    # def compute_reprojection_error(yaw_adjust):
    #     # 构建调整后的旋转矩阵
    #     adjusted_rpy = np.array([final_rpy[0], final_rpy[1], yaw_adjust])
    #     R_adjusted = euler_to_rotation_matrix(adjusted_rpy)
    #
    #     # 重建变换矩阵（位置保持不变）
    #     T_world_adj = np.eye(4)
    #     T_world_adj[:3, :3] = R_adjusted
    #     T_world_adj[:3, 3] = final_pos
    #
    #     # 计算从世界到相机的变换
    #     # T_camera = inv(T_p2h) @ inv(T_h2w) @ T_world_adj
    #     # 但更简单：直接从装甲板到相机
    #     T_camera = np.linalg.inv(T_p2h @ T_h2w) @ T_world_adj
    #
    #     # 将3D点变换到相机坐标系
    #     points_camera = []
    #     for p in obj_pts:
    #         p_homo = np.append(p, 1.0)
    #         p_cam = T_camera @ p_homo
    #         points_camera.append(p_cam[:3])
    #
    #     # 投影到图像平面
    #     total_error = 0
    #     for p_cam, p_img in zip(points_camera, img_pts):
    #         # 投影
    #         p_proj_homo = camera_k @ p_cam
    #         p_proj = p_proj_homo[:2] / p_proj_homo[2]
    #
    #         # 计算误差
    #         total_error += np.linalg.norm(p_proj - p_img)
    #
    #     return total_error / len(obj_pts)
    #
    # # 3. 在基准yaw附近搜索最优yaw
    # search_range = 0.3  # 弧度，约±17°
    # best_yaw = base_yaw
    # best_error = compute_reprojection_error(base_yaw)
    #
    # # 简单网格搜索
    # for adjust in np.linspace(-search_range, search_range, 61):
    #     test_yaw = base_yaw + adjust
    #     error = compute_reprojection_error(test_yaw)
    #     if error < best_error:
    #         best_error = error
    #         best_yaw = test_yaw
    #
    # # 4. 返回优化后的结果
    # final_rpy_optimized = np.array([final_rpy[0], final_rpy[1], best_yaw])
    # return final_pos, final_rpy_optimized

    # 6. 增强PnP（YawPnP优化）- 修正版
    solver = YawPnP(camera_k)

    # 关键修正1：sys_yaw设置为装甲板在世界坐标系中的yaw
    solver.sys_yaw = final_rpy[2]

    # 关键修正2：pose设置为装甲板在世界坐标系中的位置
    solver.pose = np.append(final_pos, 1.0)

    # 关键修正3：T应该是世界坐标系到相机坐标系的变换
    # T_camera = inv(T_p2h @ T_h2w)
    T_camera = np.linalg.inv(T_p2h @ T_h2w)
    solver.T = T_camera
    solver.T_inv = np.linalg.inv(T_camera)

    # 设置3D点和2D点
    solver.set_world_points(obj_pts)
    solver.set_image_points(img_pts)

    # 设置俯仰角
    solver.set_elevation(final_rpy[1])

    # 优化（现在append_yaw是对base_yaw的微调）
    limit = 0.3  # 缩小搜索范围
    y_pix = solver.ternary_search('pixel', -limit, limit)
    y_ang = solver.ternary_search('angle', -limit, limit)

    # 混合结果
    mid, length = 0.15, 0.05  # 缩小范围
    abs_y = abs(y_pix)

    if (mid - length / 2) < abs_y < (mid + length / 2):
        ratio = 0.5 + 0.5 * math.sin(math.pi * (abs_y - mid) / length)
        append_yaw = ratio * y_pix + (1 - ratio) * y_ang
    elif abs_y <= (mid - length / 2):
        append_yaw = y_ang
    else:
        append_yaw = y_pix

    # 应用修正
    optimized_yaw = final_rpy[2] + append_yaw
    final_rpy_optimized = np.array([final_rpy[0], final_rpy[1], optimized_yaw])

    return final_pos, final_rpy_optimized

