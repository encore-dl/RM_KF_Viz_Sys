from collections import deque


class SlideWeightedAvg:
    """加权滑动平均"""

    def __init__(self, size: int = 500):
        self.size = size
        self.values = deque()
        self.weights = deque()
        self.sum_values = 0.0
        self.sum_weights = 0.0

    def push(self, value: float, weight: float):
        if len(self.values) >= self.size:
            old_value = self.values.popleft()
            old_weight = self.weights.popleft()
            self.sum_values -= old_value * old_weight
            self.sum_weights -= old_weight

        self.values.append(value)
        self.weights.append(weight)
        self.sum_values += value * weight
        self.sum_weights += weight

    def get_avg(self) -> float:
        if self.sum_weights == 0:
            return 0.0
        return self.sum_values / self.sum_weights

    def get_size(self) -> int:
        return len(self.values)

