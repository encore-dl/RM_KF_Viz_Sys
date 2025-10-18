import time
import math
import numpy as np

from RM_KF_Viz_Sys.object.Armor import ArmorName
from RM_KF_Viz_Sys.object.Target import Target
from RM_KF_Viz_Sys.tools.EKF_rel.EKF import ExtendedKalmanFilter

from RM_KF_Viz_Sys.tools.EKF_rel.EKF_tools import limit_rad

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800

class Tracker:
    """跟踪器 - 严格遵循源码结构"""

    def __init__(self):
        self.state_ = "lost"
        self.pre_state_ = "lost"

        # 状态机参数
        self.detect_count_ = 0
        self.min_detect_count_ = 5
        self.temp_lost_count_ = 0
        self.max_temp_lost_count_ = 15
        self.normal_temp_lost_count_ = 15
        self.outpost_max_temp_lost_count_ = 75

        self.last_timestamp_ = time.time()
        self.target_ = None

    def state_machine(self, found):
        """状态机 - 严格遵循源码逻辑"""
        if self.state_ == "lost":
            if not found:
                return
            # print("someone's detecting you o> ")
            self.state_ = "detecting"
            self.detect_count_ = 1

        elif self.state_ == "detecting":
            if found:
                self.detect_count_ += 1
                if self.detect_count_ >= self.min_detect_count_:
                    self.state_ = "tracking"
            else:
                self.detect_count_ = 0
                self.state_ = "lost"

        elif self.state_ == "tracking":
            if not found:
                self.temp_lost_count_ = 1
                self.state_ = "temp_lost"

        elif self.state_ == "switching":
            if found:
                self.state_ = "detecting"
            else:
                self.temp_lost_count_ += 1
                if self.temp_lost_count_ > 200:
                    self.state_ = "lost"

        elif self.state_ == "temp_lost":
            if found:
                self.state_ = "tracking"
            else:
                self.temp_lost_count_ += 1

                # 前哨站特殊处理
                if self.target_.name == ArmorName.OUTPOST:
                    self.max_temp_lost_count_ = self.outpost_max_temp_lost_count_
                else:
                    self.max_temp_lost_count_ = self.normal_temp_lost_count_

                if self.temp_lost_count_ > self.max_temp_lost_count_:
                    self.state_ = "lost"

    def set_target(self, armors, t):
        """设置新目标 - 简化版本"""
        if not armors:
            return False

        armor = armors[0]

        # 初始化目标
        radius = 0.2  # 默认半径
        armor_num = 4  # 默认4个装甲板
        P0_dig = np.array([1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1])

        self.target_ = Target(radius, armor_num)
        self.target_.name = armor.name
        self.target_.armor_type = armor.type
        self.target_.priority = armor.priority
        self.target_.t_ = t

        # 初始化EKF状态
        xyz = armor.xyz_in_world
        ypr = armor.ypr_in_world

        # 计算旋转中心 (遵循源码逻辑)
        center_x = xyz[0] + radius * math.cos(ypr[0])
        center_y = xyz[1] + radius * math.sin(ypr[0])
        center_z = xyz[2]

        # 初始化状态向量
        x0 = np.array([
            center_x, 0, center_y, 0, center_z, 0,  # 位置和速度
            ypr[0], 0,  # 角度和角速度
            radius, 0, 0  # 半径、长短轴差、高度差
        ])

        P0 = np.diag(P0_dig)

        # 角度加法函数
        def x_add(a, b):
            c = a + b
            c[6] = limit_rad(c[6])  # 限制角度
            return c

        self.target_.ekf_ = ExtendedKalmanFilter(x0, P0, x_add)

        return True

    def update_target(self, armors, t):
        """更新目标 - 严格遵循源码接口"""
        if not self.target_:
            return False

        # 预测
        self.target_.predict(t)

        # 统计匹配的装甲板数量
        found_count = 0
        min_x = float('inf')
        for armor in armors:
            if (armor.name != self.target_.name or
                    armor.type != self.target_.armor_type):
                continue
            found_count += 1
            min_x = min(min_x, armor.xyz_in_world[0])

        if found_count == 0:
            return False

        # 更新目标
        for armor in armors:
            if (armor.name != self.target_.name or
                    armor.type != self.target_.armor_type):
                continue

            # 这里在真实系统中会调用solver.solve(armor)
            # 在模拟中，我们假设armor已经包含正确的3D坐标
            self.target_.update(armor)
            break  # 只更新第一个匹配的装甲板

        return True

    def track(self, armors, t):
        """主跟踪函数 - 简化版本"""
        dt = t - self.last_timestamp_
        self.last_timestamp_ = t

        # 时间间隔检测
        if self.state_ != "lost" and dt > 0.1:
            print(f"Large dt: {dt:.3f}s")
            self.state_ = "lost"

        # 装甲板排序 (简化)
        if armors:
            # 按距离图像中心排序
            img_center = np.array([SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2])
            armors.sort(key=lambda a: np.linalg.norm(
                np.array([a.xyz_in_world[0] * 1000, a.xyz_in_world[1] * 1000]) - img_center))

            # 按优先级排序
            armors.sort(key=lambda a: a.priority)

        # 目标跟踪
        found = False
        if self.state_ == "lost":
            found = self.set_target(armors, t)
        else:
            found = self.update_target(armors, t)

        # 更新状态机
        self.pre_state_ = self.state_
        self.state_machine(found)

        # 发散检测
        if self.state_ != "lost" and self.target_.diverged():
            print("Target diverged!")
            self.state_ = "lost"
            return []

        # 收敛检测
        if (self.state_ != "lost" and
                self.target_.ekf().data["recent_nis_failures"] >=
                0.4 * self.target_.ekf().window_size):
            print("Bad convergence!")
            self.state_ = "lost"
            return []

        if self.state_ == "lost":
            return []

        return [self.target_]