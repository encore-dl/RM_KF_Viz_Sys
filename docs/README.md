# RoboMaster 通用预测器接口模拟器

本项目是一个用于开发和测试 RoboMaster 机器人跟踪与预测算法的可视化模拟环境。它提供完整的物理仿真、相机成像、PnP 解算以及多目标跟踪框架，旨在帮助算法工程师快速验证滤波器、运动模型与决策逻辑。

## 快速开始

### 环境要求
- Python 3.8+
- NumPy, SciPy, Pygame, PyYAML

### 安装依赖
```bash
pip install numpy scipy pygame pyyaml
```

### 运行模拟器
```bash
python main.py
```

### 控制说明
| 按键 | 功能 |
|------|------|
| 方向键 | 平移选中实体 |
| A/D   | 缓慢旋转（偏航） |
| Z/C   | 快速旋转（偏航） |
| W/X   | 俯仰控制 |
| 空格  | 急停 |
| 数字键1~9 | 选择对应机器人（需先创建） |
| 9 (小键盘) | 切换相机自动瞄准 |
| R     | 重置模拟器 |
| ESC   | 退出 |

## 项目结构
```
simulator/
├── core/               # 核心实体与算法
├── models/             # 跟踪器模型（实现统一接口）
├── simulation/         # 模拟器运行与管理
├── config/             # 配置文件
├── utils/              # 通用工具
└── main.py             # 入口
```

详细文档请参考 `specs/` 目录下的各项规范。