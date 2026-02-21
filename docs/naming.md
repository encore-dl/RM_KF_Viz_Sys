# 命名规范

为了保持代码一致性与可读性，本项目遵循以下命名规则。

## 通用原则
- 命名应具有描述性，避免含糊缩写。
- 优先使用英文全称，局部作用域允许合理缩写（如 `idx`、`tmp`）。
- 遵循 Python PEP 8 风格指南。

## 命名风格
| 类型               | 风格               | 示例                     |
|--------------------|--------------------|--------------------------|
| 模块（文件名）      | 小写+下划线        | `kalman_filter.py`       |
| 包                 | 小写，无下划线     | `core`, `models`         |
| 类                 | 驼峰（首字母大写） | `KalmanFilter`, `Robot`  |
| 函数/方法          | 小写+下划线        | `predict()`, `get_obs()` |
| 变量               | 小写+下划线        | `obs_armors`, `dt`       |
| 常量               | 全大写+下划线      | `MAX_SPEED`, `PI`        |
| 私有成员（类内）    | 前导下划线         | `_internal_func`         |
| 保护成员（子类可用）| 单下划线           | `_protected_attr`        |

## 缩写建议
| 全称           | 允许缩写 | 示例                     |
|----------------|----------|--------------------------|
| observation    | obs      | `obs_armors`             |
| prediction     | pred     | `pred_pos`               |
| manager        | mgr      | `robot_mgr`              |
| camera         | cam      | `cam_mgr`                |
| configuration  | cfg      | `cfg.track_queue_size`   |
| timestamp      | ts       | `obs.ts`                 |
| position       | pos      | `robot.pos`              |
| orientation    | rpy      | `camera.rpy` (roll,pitch,yaw) |
| velocity       | vel      | `center.vel`             |
| angular_velocity | omg    | `robot.omg`              |

## 特殊约定
- 状态向量索引：使用大写常量命名，如 `IDX_X = 0`, `IDX_VX = 3`。
- 滤波器相关：`x` 表示状态向量，`P` 表示协方差，`F` 为状态转移矩阵，`H` 为观测矩阵。
- 时间相关：`t` 表示绝对时间戳，`dt` 表示时间间隔。