import numpy as np
from collections import deque
import threading
from typing import Tuple


class SlideIntegrator:
    """滑动积分器，用于自身运动补偿"""

    def __init__(self, size: int = 200, delay: float = 0.1):
        self.size = size
        self.delay = delay
        self.values = deque()
        self.mutex = threading.Lock()

    def push(self, v_magnitude: float, v_angle: float, timestamp: float):
        with self.mutex:
            self.values.append({
                'v_magnitude': v_magnitude,
                'v_angle': v_angle,
                'timestamp': timestamp
            })

            if len(self.values) > self.size:
                self.values.popleft()

            # 移除过时的数据
            while self.values and (timestamp - self.values[0]['timestamp']) > self.delay:
                self.values.popleft()

    def get_integral(self, start_time: float, end_time: float) -> Tuple[float, float]:
        with self.mutex:
            if not self.values:
                return 0.0, 0.0

            if start_time > end_time:
                return 0.0, 0.0

            x_integral = 0.0
            y_integral = 0.0

            # 找到第一个时间点大于等于start_time的值
            values_list = list(self.values)
            start_idx = 0
            for i, value in enumerate(values_list):
                if value['timestamp'] >= start_time:
                    start_idx = i
                    break

            current_time = start_time
            for i in range(start_idx, len(values_list)):
                value = values_list[i]
                if value['timestamp'] > end_time:
                    dt = end_time - current_time
                    x_integral += value['v_magnitude'] * np.cos(value['v_angle']) * dt
                    y_integral += value['v_magnitude'] * np.sin(value['v_angle']) * dt
                    break
                else:
                    dt = value['timestamp'] - current_time
                    x_integral += value['v_magnitude'] * np.cos(value['v_angle']) * dt
                    y_integral += value['v_magnitude'] * np.sin(value['v_angle']) * dt
                    current_time = value['timestamp']

            return x_integral, y_integral

    def get_size(self) -> int:
        return len(self.values)

    def clear(self):
        with self.mutex:
            self.values.clear()