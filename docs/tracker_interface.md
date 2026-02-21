# 跟踪器接口规范

所有具体的跟踪器必须实现 `TrackerBase` 抽象基类，以保证与模拟器其他模块的无缝协作。

## 1. 基础数据结构

### `ArmorObservation`
```python
@dataclass
class ArmorObservation:
    armor_id: int          # 装甲板编号（0~3）
    robot_type: RobotType  # 机器人类型（枚举）
    pos: np.ndarray        # 世界坐标系位置 (3,)
    yaw: float             # 世界坐标系偏航角 (弧度)
    size: str              # 'large' 或 'small'
```

### `Observation`
```python
@dataclass
class Observation:
    armors: List[ArmorObservation]  # 当前帧观测到的所有装甲板
    timestamp: float                 # 观测时间戳（秒）
```

### `Prediction`
```python
@dataclass
class Prediction:
    center: np.ndarray        # 预测的机器人中心位置 (3,)
    armors: List[np.ndarray]  # 预测的各个装甲板位置，长度等于装甲板数量
    timestamp: float          # 预测对应的时间戳
    is_tracking: bool         # 是否正在跟踪
```

## 2. 跟踪器基类 `TrackerBase`

```python
from abc import ABC, abstractmethod
from typing import List, Optional

class TrackerBase(ABC):
    @abstractmethod
    def push_observation(self, obs: Observation) -> None:
        """输入一帧观测数据，触发内部预测与更新"""
        pass

    @abstractmethod
    def get_prediction(self, fly_time: float) -> Optional[Prediction]:
        """
        获取指定飞行时间后的预测结果。
        :param fly_time: 子弹飞行时间（秒），从当前时刻开始计算。
        :return: 预测结果，若未跟踪则返回 None。
        """
        pass

    @property
    @abstractmethod
    def is_tracking(self) -> bool:
        """返回当前是否处于跟踪状态"""
        pass
```

## 3. 实现要求
- 跟踪器内部可包含多个子滤波器（如卡尔曼、UKF、IMM），但对外只需暴露上述接口。
- 所有角度运算必须使用 `limit_rad` 和 `safe_angle_sub` 处理缠绕。
- 状态向量索引应使用有意义的常量，例如：
  ```python
  IDX_X, IDX_Y, IDX_Z = 0, 1, 2
  IDX_VX, IDX_VY, IDX_VZ = 3, 4, 5
  IDX_PSI, IDX_W = 6, 7   # 偏航角及角速度
  ```
- 初始化时机：当首次收到观测且 `is_tracking == False` 时，应执行初始化。
- 丢失处理：超过一定时间无观测应自动重置跟踪状态。

## 4. 示例（伪代码）
```python
class MyTracker(TrackerBase):
    def __init__(self, config):
        self.kf = KalmanFilter(...)
        self.last_t = None
        self._is_tracking = False

    def push_observation(self, obs: Observation):
        if not self._is_tracking:
            self._init(obs)
        else:
            dt = obs.timestamp - self.last_t
            self.kf.predict(dt)
            self.kf.update(obs)
        self.last_t = obs.timestamp

    def get_prediction(self, fly_time: float) -> Optional[Prediction]:
        if not self._is_tracking:
            return None
        pred_center = self.kf.x[:3] + self.kf.x[3:6] * fly_time
        # ... 计算装甲板位置
        return Prediction(center=pred_center, armors=..., timestamp=time.time(), is_tracking=True)

    @property
    def is_tracking(self):
        return self._is_tracking
```

## 5. 与模拟器的集成
- 模拟器中的 `TrackerManager` 负责在独立线程中调用跟踪器的 `push_observation`，并定期通过 `get_prediction` 获取结果供可视化。
- 跟踪器不应包含任何与 Pygame 或可视化相关的代码，保持纯粹的逻辑。