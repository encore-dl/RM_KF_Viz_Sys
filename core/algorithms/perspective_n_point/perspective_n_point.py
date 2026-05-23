import cv2
import numpy as np
import math
from dataclasses import dataclass
from typing import Optional, Tuple

# 假设 rotation_matrix_to_euler 已在 core.algorithms.math 中定义
# 下面是一个简单示例，实际使用时请替换为你的实现
def rotation_matrix_to_euler(R):
    """从旋转矩阵提取欧拉角 (roll, pitch, yaw)"""
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])

# ==================== 配置类 ====================
@dataclass
class PnPConfig:
    width_large: float = 0.230
    height_large: float = 0.055
    width_small: float = 0.135
    height_small: float = 0.055


# ==================== 自定义YawPnP求解器 ====================
class YawPnP:
    def __init__(self, intrinsic_matrix, verbose=False):
        self.Kc = intrinsic_matrix
        self.verbose = verbose
        self.P_world = []          # 世界FLU点（内部存储格式 [0, -y, -z]）
        self.P_pixel = []          # 像素坐标
        self.sys_yaw = 0.0
        self.pose = np.zeros(4)
        self.T = np.eye(4)         # 世界 -> 相机RDF 变换矩阵
        self.T_inv = np.eye(4)
        self.elevation = 0
        self.ANGLE_UP_15 = 15.0 * math.pi / 180.0
        self.ANGLE_UP_75 = 75.0 * math.pi / 180.0
        self.ANGLE_DOWN_15 = -15.0 * math.pi / 180.0
        self.prior_yaw = 0.0       # 先验偏航角（来自IPPE或上一帧）

    def set_world_points(self, object_points):
        self.P_world = []
        for p in object_points:
            self.P_world.append(np.array([0, -p[1], -p[2], 1.0]))

    def set_image_points(self, image_points):
        self.P_pixel = [p for p in image_points]

    def set_elevation(self, pitch_rad):
        if pitch_rad > 0.5:
            self.elevation = 1
        elif pitch_rad < -0.2:
            self.elevation = 2
        else:
            self.elevation = 0

    def get_w_h_from_world_points(self):
        ys = [p[1] for p in self.P_world]
        zs = [p[2] for p in self.P_world]
        w = max(ys) - min(ys)
        h = max(zs) - min(zs)
        return w, h

    def compute_edge_direction(self, idx_a, idx_b):
        p_a = self.P_pixel[idx_a]
        p_b = self.P_pixel[idx_b]
        dx = p_b[0] - p_a[0]
        dy = p_b[1] - p_a[1]
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            return 0.0, 0.0
        return dx / norm, dy / norm

    def _pqr_from_UVW(self, U3d, V3d, W3d, u, v):
        """从三维向量 U,V,W 和观测方向 (u,v) 计算 p,q,r"""
        U2d = np.array([-U3d[1], U3d[0]])
        V2d = np.array([-V3d[1], V3d[0]])
        W2d = np.array([-W3d[1], W3d[0]])
        p = U2d[0] * v - U2d[1] * u
        q = V2d[0] * v - V2d[1] * u
        r = W2d[0] * v - W2d[1] * u
        return p, q, r

    # ------------------------------------------------------------
    # 解析法：仅使用两条水平边（上边和下边），正则化求解
    # ------------------------------------------------------------
    def solve_analytic_horizontal_only(self):
        w, h = self.get_w_h_from_world_points()
        if w < 1e-6 or h < 1e-6:
            return self.prior_yaw
        H = h / 2.0

        t_world = self.pose[:3]
        R_w2c = self.T[:3, :3]
        t_w2c = self.T[:3, 3]
        t_cam = R_w2c @ t_world + t_w2c

        if self.elevation == 0:
            theta = self.ANGLE_UP_15
        elif self.elevation == 1:
            theta = self.ANGLE_UP_75
        else:
            theta = self.ANGLE_DOWN_15
        c_theta = math.cos(theta)
        s_theta = math.sin(theta)

        c1 = R_w2c[:, 0]
        c2 = R_w2c[:, 1]
        c3 = R_w2c[:, 2]

        def cross(a, b):
            return np.array([a[1]*b[2] - a[2]*b[1],
                             a[2]*b[0] - a[0]*b[2],
                             a[0]*b[1] - a[1]*b[0]])

        d1 = cross(c1, t_cam)
        d2 = cross(c2, t_cam)

        u_top, v_top = self.compute_edge_direction(0, 1)  # 上边 (索引0->1)
        u_bot, v_bot = self.compute_edge_direction(2, 3)  # 下边 (索引2->3)

        # 上边 (Z=H)
        U_top =  H * c_theta * c1 - d2
        V_top =  H * c_theta * c2 + d1
        W_top = -H * s_theta * c3
        p1, q1, r1 = self._pqr_from_UVW(U_top, V_top, W_top, u_top, v_top)

        # 下边 (Z=-H)
        U_bot = -H * c_theta * c1 - d2
        V_bot = -H * c_theta * c2 + d1
        W_bot =  H * s_theta * c3
        p2, q2, r2 = self._pqr_from_UVW(U_bot, V_bot, W_bot, u_bot, v_bot)

        if self.verbose:
            print("========== 水平边诊断 ==========")
            print(f"t_cam = [{t_cam[0]:.3f}, {t_cam[1]:.3f}, {t_cam[2]:.3f}]")
            print(f"上边 p={p1:.6f} q={q1:.6f} r={r1:.6f}  ratio={abs(r1)/(abs(p1)+abs(q1)+1e-9):.4f}")
            print(f"下边 p={p2:.6f} q={q2:.6f} r={r2:.6f}  ratio={abs(r2)/(abs(p2)+abs(q2)+1e-9):.4f}")
            c_ref = math.cos(self.prior_yaw)
            s_ref = math.sin(self.prior_yaw)
            D1 = q1 * c_ref - p1 * s_ref
            D2 = q2 * c_ref - p2 * s_ref
            print(f"D_top (ref) = {D1:.6f}, D_bot (ref) = {D2:.6f}")
            # 预测方向（使用先验 yaw）
            U1 = U_top; V1 = V_top; W1 = W_top
            n_pred1 = U1 * c_ref + V1 * s_ref + W1
            print(f"上边预测方向 (prior) = [{n_pred1[0]:.4f}, {n_pred1[1]:.4f}]")
            print(f"上边观测方向 = [{u_top:.4f}, {v_top:.4f}]")

        # 构建方程组
        A = np.array([[p1, q1], [p2, q2]], dtype=np.float64)
        b = -np.array([r1, r2], dtype=np.float64)
        lam = 5e-3
        prior_cos = math.cos(self.prior_yaw)
        prior_sin = math.sin(self.prior_yaw)
        x_prior = np.array([prior_cos, prior_sin])
        ATA = A.T @ A
        ATb = A.T @ b
        x = np.linalg.solve(ATA + lam * np.eye(2), ATb + lam * x_prior)
        norm = np.linalg.norm(x)
        if norm > 1e-6:
            x = x / norm
        else:
            x = x_prior
        # 方向约束
        to_observer = -t_world[:2]
        dist = np.linalg.norm(to_observer)
        if dist > 1e-6:
            to_obs_unit = to_observer / dist
            if np.dot(x, to_obs_unit) < 0:
                x = -x
        psi = math.atan2(x[1], x[0])
        if self.verbose:
            print(f"水平边求解结果: psi = {psi:.4f} rad ({math.degrees(psi):.2f}°)")
        return psi

    # ------------------------------------------------------------
    # 解析法：仅使用两条竖直边（左边和右边），正则化求解
    # ------------------------------------------------------------
    def solve_analytic_vertical_only(self):
        w, h = self.get_w_h_from_world_points()
        if w < 1e-6 or h < 1e-6:
            return self.prior_yaw
        W = w / 2.0

        t_world = self.pose[:3]
        R_w2c = self.T[:3, :3]
        t_w2c = self.T[:3, 3]
        t_cam = R_w2c @ t_world + t_w2c

        if self.elevation == 0:
            theta = self.ANGLE_UP_15
        elif self.elevation == 1:
            theta = self.ANGLE_UP_75
        else:
            theta = self.ANGLE_DOWN_15
        c_theta = math.cos(theta)
        s_theta = math.sin(theta)

        c1 = R_w2c[:, 0]
        c2 = R_w2c[:, 1]
        c3 = R_w2c[:, 2]

        def cross(a, b):
            return np.array([a[1]*b[2] - a[2]*b[1],
                             a[2]*b[0] - a[0]*b[2],
                             a[0]*b[1] - a[1]*b[0]])

        # 预计算常量
        a1 = c_theta * c1
        b1 = c_theta * c2
        d1 = -s_theta * c3
        a3 = s_theta * c1
        b3 = s_theta * c2
        d3 = c_theta * c3

        a3_cross_t = cross(a3, t_cam)
        b3_cross_t = cross(b3, t_cam)
        d3_cross_t = cross(d3, t_cam)

        u_r, v_r = self.compute_edge_direction(1, 2)  # 右边 (索引1->2)
        u_l, v_l = self.compute_edge_direction(3, 0)  # 左边 (索引3->0)

        def compute_coeffs(Y_V, u_hat, v_hat):
            U3d = -Y_V * a1 + a3_cross_t
            V3d = -Y_V * b1 + b3_cross_t
            W3d = -Y_V * d1 + d3_cross_t
            U2d = np.array([-U3d[1], U3d[0]])
            V2d = np.array([-V3d[1], V3d[0]])
            W2d = np.array([-W3d[1], W3d[0]])
            p = U2d[0]*v_hat - U2d[1]*u_hat
            q = V2d[0]*v_hat - V2d[1]*u_hat
            r = W2d[0]*v_hat - W2d[1]*u_hat
            return p, q, r, U2d, V2d, W2d

        p_r, q_r, r_r, U_r, V_r, W_r = compute_coeffs(-W, u_r, v_r)  # Y_V = -W
        p_l, q_l, r_l, U_l, V_l, W_l = compute_coeffs( W, u_l, v_l)  # Y_V =  W

        if self.verbose:
            print("========== 竖直边诊断 ==========")
            print(f"t_cam = [{t_cam[0]:.3f}, {t_cam[1]:.3f}, {t_cam[2]:.3f}]")
            print(f"右边 p={p_r:.6f} q={q_r:.6f} r={r_r:.6f}  ratio={abs(r_r)/(abs(p_r)+abs(q_r)+1e-9):.4f}")
            print(f"左边 p={p_l:.6f} q={q_l:.6f} r={r_l:.6f}  ratio={abs(r_l)/(abs(p_l)+abs(q_l)+1e-9):.4f}")
            c_ref = math.cos(self.prior_yaw)
            s_ref = math.sin(self.prior_yaw)
            D_r = q_r * c_ref - p_r * s_ref
            D_l = q_l * c_ref - p_l * s_ref
            print(f"D_right (ref) = {D_r:.6f}, D_left (ref) = {D_l:.6f}")
            # 预测方向（使用先验 yaw）
            n_r_pred = U_r * c_ref + V_r * s_ref + W_r
            n_l_pred = U_l * c_ref + V_l * s_ref + W_l
            print(f"右边预测方向 (prior) = [{n_r_pred[0]:.4f}, {n_r_pred[1]:.4f}]")
            print(f"左边预测方向 (prior) = [{n_l_pred[0]:.4f}, {n_l_pred[1]:.4f}]")
            print(f"右边观测方向 = [{u_r:.4f}, {v_r:.4f}]")
            print(f"左边观测方向 = [{u_l:.4f}, {v_l:.4f}]")

        # 构建方程组
        A = np.array([[p_r, q_r], [p_l, q_l]], dtype=np.float64)
        b = -np.array([r_r, r_l], dtype=np.float64)
        lam = 5e-3
        prior_cos = math.cos(self.prior_yaw)
        prior_sin = math.sin(self.prior_yaw)
        x_prior = np.array([prior_cos, prior_sin])
        ATA = A.T @ A
        ATb = A.T @ b
        x = np.linalg.solve(ATA + lam * np.eye(2), ATb + lam * x_prior)
        norm = np.linalg.norm(x)
        if norm > 1e-6:
            x = x / norm
        else:
            x = x_prior
        # 方向约束
        to_observer = -t_world[:2]
        dist = np.linalg.norm(to_observer)
        if dist > 1e-6:
            to_obs_unit = to_observer / dist
            if np.dot(x, to_obs_unit) < 0:
                x = -x
        psi = math.atan2(x[1], x[0])
        if self.verbose:
            print(f"竖直边求解结果: psi = {psi:.4f} rad ({math.degrees(psi):.2f}°)")
        return psi


# ==================== 主求解函数 ====================
def solve_pnp_core(camera_k, camera_dist, armor_size, image_points_2d,
                   head_yaw, trans_head2world, rot_head2world,
                   trans_pnp2head, rot_pnp2head, use_plus_pnp=True):
    """
    PnP求解核心函数
    返回 (final_pos, final_rpy)   final_rpy = [roll, pitch, yaw]
    其中 yaw 是世界绝对偏航角（与观测车无关）
    """
    cfg = PnPConfig()
    if armor_size == 'large':
        w, h = cfg.width_large, cfg.height_large
    else:
        w, h = cfg.width_small, cfg.height_small

    obj_pts = np.array([
        [0,  w/2,  h/2],
        [0, -w/2,  h/2],
        [0, -w/2, -h/2],
        [0,  w/2, -h/2]
    ], dtype=np.float64)

    img_pts = np.ascontiguousarray(image_points_2d).astype(np.float64)

    # 标准 IPPE 求解
    success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, camera_k, camera_dist,
                                       flags=cv2.SOLVEPNP_IPPE)
    if not success or np.isnan(rvec).any() or np.isnan(tvec).any():
        return None, None

    rot_pnp, _ = cv2.Rodrigues(rvec)

    T_pnp = np.eye(4)
    T_pnp[:3, :3] = rot_pnp
    T_pnp[:3, 3] = tvec.flatten()

    T_p2h = np.eye(4)
    T_p2h[:3, :3] = rot_pnp2head
    T_p2h[:3, 3] = trans_pnp2head

    T_h2w = np.eye(4)
    T_h2w[:3, :3] = rot_head2world
    T_h2w[:3, 3] = trans_head2world

    T_world = T_h2w @ T_p2h @ T_pnp
    final_pos = T_world[:3, 3]
    if np.isnan(final_pos).any():
        return None, None

    rot_world = T_world[:3, :3]
    final_rpy = rotation_matrix_to_euler(rot_world)   # [roll, pitch, yaw] 世界绝对角

    if not use_plus_pnp:
        return final_pos, final_rpy

    # ========== 解析法优化 ==========
    solver = YawPnP(camera_k, verbose=True)   # 开启诊断输出
    solver.prior_yaw = final_rpy[2]
    solver.sys_yaw = final_rpy[2]
    solver.pose = np.append(final_pos, 1.0)

    # 构造世界 -> 相机RDF 变换矩阵
    R_camFLU_to_world = rot_head2world
    t_camFLU_to_world = trans_head2world
    R_world_to_camFLU = R_camFLU_to_world.T
    t_world_to_camFLU = -R_world_to_camFLU @ t_camFLU_to_world
    R_FLU_to_RDF = np.array([[0, -1, 0],
                             [0, 0, -1],
                             [1, 0, 0]])   # 来自 camera.R_body_to_optical
    R_world_to_camRDF = R_FLU_to_RDF @ R_world_to_camFLU
    t_world_to_camRDF = R_FLU_to_RDF @ t_world_to_camFLU

    solver.T = np.eye(4)
    solver.T[:3, :3] = R_world_to_camRDF
    solver.T[:3, 3] = t_world_to_camRDF
    solver.T_inv = np.linalg.inv(solver.T)

    solver.set_world_points(obj_pts)
    solver.set_image_points(img_pts)
    solver.set_elevation(final_rpy[1])

    # 选择一种方法（可以切换调试）
    optimized_yaw = solver.solve_analytic_horizontal_only()
    # optimized_yaw = solver.solve_analytic_vertical_only()   # 当前启用竖直边诊断
    final_rpy_optimized = np.array([final_rpy[0], final_rpy[1], optimized_yaw])
    print(f"Optimized yaw: {optimized_yaw:.4f} rad")
    return final_pos, final_rpy_optimized