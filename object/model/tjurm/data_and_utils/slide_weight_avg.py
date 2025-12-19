from collections import deque
import math


class SlideWeightedAvg:
    """加权滑动平均 (改进版)"""

    def __init__(self, size: int = 20):
        self.size = size
        self.values = deque()
        self.weights = deque()
        self.sum_values = 0.0
        self.sum_weights = 0.0
        self.avg = 0.0

    def push(self, value: float, weight: float):
        """添加值和权重"""
        # 检查NaN
        if math.isnan(value) or math.isnan(weight):
            raise ValueError("value or weight is NaN")

        # 队列已满，移除最旧的值
        if len(self.values) >= self.size:
            old_value = self.values.popleft()
            old_weight = self.weights.popleft()
            self.sum_values -= old_value * old_weight
            self.sum_weights -= old_weight

        # 添加新值
        self.values.append(value)
        self.weights.append(weight)
        self.sum_values += value * weight
        self.sum_weights += weight

        # 计算平均值
        if self.sum_weights != 0:
            self.avg = self.sum_values / self.sum_weights

    def get_avg(self) -> float:
        """获取加权平均值"""
        return self.avg

    def get_size(self) -> int:
        """返回当前队列大小"""
        return len(self.values)

    def clear(self):
        """清空所有数据"""
        self.values.clear()
        self.weights.clear()
        self.sum_values = 0.0
        self.sum_weights = 0.0
        self.avg = 0.0