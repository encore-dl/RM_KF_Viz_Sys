import math
import numpy as np


def limit_rad(angle: float) -> float:
    """限制角度在[-pi, pi]范围内"""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def get_weight_by_theta(theta: float) -> float:
    """根据角度获取权重"""
    weight = math.exp(-theta * theta * 400)
    if math.isfinite(weight):
        return weight
    return float(np.finfo(np.float64).tiny)


# 辅助函数
def safe_angle_sub(angle1: float, angle2: float) -> float:
    """安全的角度减法"""
    angle = angle1 - angle2
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def get_angle_trans(armor_count, target_angle: float, src_angle: float, refer_angle: float = None) -> float:
    """将模型内角度转换为接近新角度"""
    if refer_angle is None:
        refer_angle = src_angle

    dst_angle = src_angle

    # 调整参考角度
    while safe_angle_sub(refer_angle, target_angle) > (math.pi / armor_count):
        refer_angle -= (2 * math.pi) / armor_count
        dst_angle -= (2 * math.pi) / armor_count

    while safe_angle_sub(target_angle, refer_angle) > (math.pi / armor_count):
        refer_angle += (2 * math.pi) / armor_count
        dst_angle += (2 * math.pi) / armor_count

    return limit_rad(dst_angle)


def is_angle_trans(armor_count, target_angle: float, src_angle: float) -> bool:
    """判断是否需要角度转换"""
    differ_angle = abs(safe_angle_sub(target_angle, src_angle))
    return differ_angle > (math.pi / armor_count)


def get_toggle(armor_count, toggle, target_angle: float, src_angle: float) -> int:
    """获取切换标签"""
    if armor_count < 4:
        return 0

    differ_angle = abs(safe_angle_sub(target_angle, src_angle))
    differ_toggle = int(round(2 * differ_angle / math.pi)) % 2
    return differ_toggle ^ toggle


def get_angle_min(armor_angle, x, y, armor_count):
    center_angle = math.atan2(y, x)
    return get_angle_trans(armor_count, center_angle, armor_angle, None)


def get_distance(pose1, pose2):
    """计算3维位姿之间的距离"""
    dx = pose1[0] - pose2[0]
    dy = pose1[1] - pose2[1]
    dz = pose1[2] - pose2[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)

