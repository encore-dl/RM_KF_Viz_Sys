import random
import math
import time
import numpy as np

from RM_KF_Viz_Sys.object.Armor import Armor, ArmorName, ArmorType

PI = math.pi


def generate_armors(self):
    """生成装甲板观测数据 - 使用修正的相机投影"""
    center = self.true_target['center']
    angle = self.true_target['angle']
    radius = self.true_target['radius']
    armor_num = self.true_target['armor_num']
    armor_height = self.true_target['armor_height']

    armors = []

    for i in range(armor_num):
        # 随机检测
        if random.random() > self.detection_prob:
            continue

        # 计算装甲板真实位置
        armor_angle = angle + i * 2 * PI / armor_num

        # 长短轴区分
        use_l_h = (armor_num == 4) and (i == 1 or i == 3)
        current_radius = radius * 1.2 if use_l_h else radius
        current_z = center[2] + armor_height if use_l_h else center[2]

        armor_x = center[0] - current_radius * math.cos(armor_angle)
        armor_y = center[1] - current_radius * math.sin(armor_angle)
        armor_z = current_z

        armor_pos = [armor_x, armor_y, armor_z]

        # 使用相机视野检查
        if not self.camera.is_point_in_fov(armor_pos):
            continue

        # 创建装甲板对象
        armor = Armor()
        armor.name = ArmorName.HERO
        armor.type = ArmorType.SMALL if armor_num == 4 else ArmorType.BIG
        armor.priority = 1

        # 添加噪声的真实位置
        armor.xyz_in_world = np.array([
            armor_x + random.gauss(0, self.observation_noise_pos),
            armor_y + random.gauss(0, self.observation_noise_pos),
            armor_z + random.gauss(0, self.observation_noise_pos)
        ])

        # 计算姿态角
        armor.ypr_in_world = np.array([
            armor_angle + random.gauss(0, self.observation_noise_angle),
            0.0,  # pitch
            0.0  # roll
        ])

        # 计算角度和距离 (ypd)
        x, y, z = armor.xyz_in_world
        yaw = math.atan2(y, x)
        distance = math.sqrt(x ** 2 + y ** 2 + z ** 2)
        pitch = math.atan2(z, math.sqrt(x ** 2 + y ** 2))
        armor.ypd_in_world = np.array([yaw, pitch, distance])

        # 设置图像坐标 (使用修正的相机投影)
        pixel_pos = self.camera.world_to_pixel([-armor_x, armor_y, armor_z])
        if pixel_pos:
            armor.center = np.array(pixel_pos)
            armors.append(armor)

    return armors


def update_true_target(self, dt):
    """更新真实目标状态"""
    current_time = time.time()
    pattern_elapsed = current_time - self.pattern_start_time
    current_pattern = self.motion_patterns[self.current_pattern]

    # 检查是否需要切换模式
    if pattern_elapsed >= current_pattern["duration"]:
        self.current_pattern = (self.current_pattern + 1) % len(self.motion_patterns)
        self.pattern_start_time = current_time
        pattern_elapsed = 0
        current_pattern = self.motion_patterns[self.current_pattern]

    # 调用当前模式对应的运动函数
    motion_result = current_pattern["motion_func"](pattern_elapsed)

    # 更新目标状态
    self.true_target['center'] = motion_result['pos']
    self.true_target['vel'] = motion_result['vel']
    self.true_target['angular_vel'] = motion_result['angular_vel']

    # 更新角度
    self.true_target['angle'] += self.true_target['angular_vel'] * dt
    self.true_target['angle'] %= (2 * PI)

    # 边界检查 (在±3米范围内)
    for i in range(3):
        if abs(self.true_target['center'][i]) > 3.0:
            self.true_target['center'][i] = 3.0 * np.sign(self.true_target['center'][i])