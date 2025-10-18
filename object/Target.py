import time
import math
import numpy as np

from RM_KF_Viz_Sys.object.Armor import ArmorName
from RM_KF_Viz_Sys.object.Armor import ArmorType
from RM_KF_Viz_Sys.tools.EKF_rel.EKF import ExtendedKalmanFilter
from RM_KF_Viz_Sys.tools.EKF_rel.EKF_tools import limit_rad, xyz2ypd, xyz2ypd_jacobian

class Target:
    """目标类 - 严格遵循源码结构"""
    def __init__(self, radius=0.2, armor_num=4):
        self.name = ArmorName.INFANTRY3
        self.armor_type = ArmorType.SMALL
        self.priority = 1
        self.jumped = False
        self.last_id = 0

        # EKF相关参数
        self.armor_num_ = armor_num
        self.switch_count_ = 0
        self.update_count_ = 0
        self.is_switch_ = False
        self.is_converged_ = False

        # 初始化EKF状态
        self.radius_ = radius
        self.ekf_ = ExtendedKalmanFilter()

        # 时间戳
        self.t_ = time.time()

    def predict(self, t):
        """预测目标状态"""
        dt = t - self.t_
        self.t_ = t

        # 状态转移矩阵 (11x11)
        F = np.eye(11)
        F[0, 1] = dt  # x = x + vx*dt
        F[2, 3] = dt  # y = y + vy*dt
        F[4, 5] = dt  # z = z + vz*dt
        F[6, 7] = dt  # a = a + w*dt

        # 过程噪声 (遵循源码参数)
        v1, v2 = (10, 0.1) if self.name == ArmorName.OUTPOST else (100, 400)
        a = dt ** 4 / 4
        b = dt ** 3 / 2
        c = dt ** 2

        Q = np.zeros((11, 11))
        Q[0:2, 0:2] = [[a * v1, b * v1], [b * v1, c * v1]]  # x, vx
        Q[2:4, 2:4] = [[a * v1, b * v1], [b * v1, c * v1]]  # y, vy
        Q[4:6, 4:6] = [[a * v1, b * v1], [b * v1, c * v1]]  # z, vz
        Q[6:8, 6:8] = [[a * v2, b * v2], [b * v2, c * v2]]  # a, w

        # 前哨站转速特判
        if (self.is_converged_ and self.name == ArmorName.OUTPOST and
                abs(self.ekf_.x[7]) > 2):
            self.ekf_.x[7] = 2.51 if self.ekf_.x[7] > 0 else -2.51

        self.ekf_.predict(F, Q)

        # 限制角度范围
        self.ekf_.x[6] = limit_rad(self.ekf_.x[6])

    def update(self, armor):
        """更新目标状态"""
        # 装甲板匹配
        id = self.match_armor(armor)

        # 状态跟踪
        if id != self.last_id:
            self.is_switch_ = True
        else:
            self.is_switch_ = False

        if self.is_switch_:
            self.switch_count_ += 1

        self.last_id = id
        self.update_count_ += 1

        # 更新EKF
        self.update_ypda(armor, id)

        return True

    def match_armor(self, armor):
        """装甲板匹配算法 - 严格遵循源码逻辑"""
        min_angle_error = float('inf')
        xyza_list = self.armor_xyza_list()

        # 创建带索引的列表并按距离排序
        xyza_i_list = []
        for i, xyza in enumerate(xyza_list):
            xyza_i_list.append((xyza, i))

        # 按距离排序
        xyza_i_list.sort(key=lambda x: xyz2ypd(x[0][:3])[2])

        # 取前3个距离最小的装甲板
        best_id = 0
        for i in range(min(3, len(xyza_i_list))):
            xyza, idx = xyza_i_list[i]
            ypd = xyz2ypd(xyza[:3])

            angle_error = (abs(limit_rad(armor.ypr_in_world[0] - xyza[3])) +
                           abs(limit_rad(armor.ypd_in_world[0] - ypd[0])))

            if angle_error < min_angle_error:
                min_angle_error = angle_error
                best_id = idx

        if best_id != 0:
            self.jumped = True

        return best_id

    def update_ypda(self, armor, id):
        """Yaw-Pitch-Distance-Angle 更新 - 严格遵循源码"""
        # 观测雅可比矩阵
        H = self.h_jacobian(self.ekf_.x, id)

        # 自适应观测噪声 (遵循源码逻辑)
        center_yaw = math.atan2(armor.xyz_in_world[1], armor.xyz_in_world[0])
        delta_angle = limit_rad(armor.ypr_in_world[0] - center_yaw)

        R_dig = np.array([
            4e-3,  # yaw噪声
            4e-3,  # pitch噪声
            math.log(abs(delta_angle) + 1) + 1,  # 距离噪声
            math.log(abs(armor.ypd_in_world[2]) + 1) / 200 + 9e-2  # 角度噪声
        ])
        R = np.diag(R_dig)

        # 观测函数
        def h_func(x):
            xyz = self.h_armor_xyz(x, id)
            ypd = xyz2ypd(xyz)
            angle = limit_rad(x[6] + id * 2 * math.pi / self.armor_num_)
            return np.array([ypd[0], ypd[1], ypd[2], angle])

        # 观测值
        z = np.array([armor.ypd_in_world[0], armor.ypd_in_world[1],
                      armor.ypd_in_world[2], armor.ypr_in_world[0]])

        # 角度差计算函数
        def z_subtract(a, b):
            c = a - b
            c[0] = limit_rad(c[0])  # yaw
            c[1] = limit_rad(c[1])  # pitch
            c[3] = limit_rad(c[3])  # angle
            return c

        self.ekf_.update(z, H, R, h_func, z_subtract)

    def h_armor_xyz(self, x, id):
        """计算装甲板3D坐标 - 严格遵循源码"""
        angle = limit_rad(x[6] + id * 2 * math.pi / self.armor_num_)
        use_l_h = (self.armor_num_ == 4) and (id == 1 or id == 3)

        r = x[8] + x[9] if use_l_h else x[8]
        armor_x = x[0] - r * math.cos(angle)
        armor_y = x[2] - r * math.sin(angle)  # 注意: 源码中y是第2个状态
        armor_z = x[4] + x[10] if use_l_h else x[4]

        return np.array([armor_x, armor_y, armor_z])

    def h_jacobian(self, x, id):
        """计算观测雅可比矩阵 - 严格遵循源码"""
        angle = limit_rad(x[6] + id * 2 * math.pi / self.armor_num_)
        use_l_h = (self.armor_num_ == 4) and (id == 1 or id == 3)

        r = x[8] + x[9] if use_l_h else x[8]
        dx_da = r * math.sin(angle)
        dy_da = -r * math.cos(angle)

        dx_dr = -math.cos(angle)
        dy_dr = -math.sin(angle)
        dx_dl = -math.cos(angle) if use_l_h else 0.0
        dy_dl = -math.sin(angle) if use_l_h else 0.0

        dz_dh = 1.0 if use_l_h else 0.0

        # 装甲板坐标对状态的雅可比
        H_armor_xyza = np.array([
            [1, 0, 0, 0, 0, 0, dx_da, 0, dx_dr, dx_dl, 0],
            [0, 0, 1, 0, 0, 0, dy_da, 0, dy_dr, dy_dl, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, dz_dh],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        ])

        # 坐标到角度距离的雅可比
        armor_xyz = self.h_armor_xyz(x, id)
        H_armor_ypd = xyz2ypd_jacobian(armor_xyz)

        H_armor_ypda = np.array([
            [H_armor_ypd[0, 0], H_armor_ypd[0, 1], H_armor_ypd[0, 2], 0],
            [H_armor_ypd[1, 0], H_armor_ypd[1, 1], H_armor_ypd[1, 2], 0],
            [H_armor_ypd[2, 0], H_armor_ypd[2, 1], H_armor_ypd[2, 2], 0],
            [0, 0, 0, 1]
        ])

        return H_armor_ypda @ H_armor_xyza

    def armor_xyza_list(self):
        """获取所有装甲板的3D坐标和角度"""
        xyza_list = []
        for i in range(self.armor_num_):
            angle = limit_rad(self.ekf_.x[6] + i * 2 * math.pi / self.armor_num_)
            xyz = self.h_armor_xyz(self.ekf_.x, i)
            xyza_list.append(np.array([xyz[0], xyz[1], xyz[2], angle]))
        return xyza_list

    def diverged(self):
        """发散检测 - 严格遵循源码逻辑"""
        r_ok = 0.05 < self.ekf_.x[8] < 0.5
        l_ok = 0.05 < (self.ekf_.x[8] + self.ekf_.x[9]) < 0.5
        return not (r_ok and l_ok)

    def converged(self):
        """收敛判断 - 严格遵循源码逻辑"""
        if self.name != ArmorName.OUTPOST and self.update_count_ > 3 and not self.diverged():
            self.is_converged_ = True

        if self.name == ArmorName.OUTPOST and self.update_count_ > 10 and not self.diverged():
            self.is_converged_ = True

        return self.is_converged_

    def ekf_x(self):
        return self.ekf_.x

    def ekf(self):
        return self.ekf_

