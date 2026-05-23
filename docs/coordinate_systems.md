# 坐标系与约定

本项目严格遵循 **右手定则 (Right-Hand Rule)**。为了兼容机器人控制与计算机视觉算法，我们分别采用了 **ISO 8855 / ROS 标准**（用于物理世界）和 **OpenCV 标准**（用于相机成像）。

## 1. World / Body Frame (世界/车体坐标系)
*适用范围：物理运动、机器人位置、速度、IMU数据。*

采用 **FLU (Front-Left-Up)** 标准：

```text
                       Z (Up, +)
                       ^
                       |
                       |
                       |    X (Forward, +)
                       |   /
                       |  /
                       | /
      Y (Left, +)      |/
      <--------------- O 
```

| 轴 (Axis) | 定义 (Definition) | 正方向 (+ Direction) | 负方向 (- Direction) |
| :--- | :--- | :--- | :--- |
| **X** | 纵向 (Longitudinal) | **前 (Forward)** | 后 (Backward) |
| **Y** | 横向 (Lateral) | **左 (Left)** | 右 (Right) |
| **Z** | 垂直 (Vertical) | **上 (Up)** | 下 (Down) |

---

## 2. Camera Optical Frame (相机光学坐标系)
*适用范围：图像投影、透视变换、视觉算法。*

采用 **OpenCV 标准** (RDF: Right-Down-Forward)：

```text
    O --------------------> X (Right, +)
    |
    |    (Screen Plane)
    |
    v Y (Down, +)

    (Z 轴垂直屏幕向里，指向前方深度方向)
```

| 轴 (Axis) | 定义 (Definition) | 正方向 (+ Direction) | 备注 |
| :--- | :--- | :--- | :--- |
| **X** | 图像水平 | **右 (Right)** | 平行于图像宽度 |
| **Y** | 图像垂直 | **下 (Down)** | 平行于图像高度 |
| **Z** | 光轴/深度 | **前 (Forward)** | 距离相机的深度 |

> **注意**：从 `Body Frame` 到 `Camera Frame` 需要经过一个固定的基底变换（Basis Change）：
> *   X_cam = -Y_body
> *   Y_cam = -Z_body
> *   Z_cam = +X_body

---

## 3. Rotation / Euler Angles (旋转与欧拉角)
*适用范围：姿态描述 (Orientation)。*

旋转正方向由 **右手螺旋定则** 判定（大拇指指向轴正向，四指弯曲方向为正）。

| 角度名称 | 旋转轴 (Axis) | 正方向动作 (+ Direction) | 直观描述 |
| :--- | :--- | :--- | :--- |
| **Yaw** (偏航) | 绕 **Z** 轴 (Up) | **左转 (Turn Left)** | 逆时针旋转 |
| **Pitch** (俯仰) | 绕 **Y** 轴 (Left) | **低头 (Nose Down)** | *注：因为Y轴朝左，右手定则导致“低头”为正* |
| **Roll** (滚转) | 绕 **X** 轴 (Forward) | **右倾 (Tilt Right)** | 顺时针侧翻 (从后方看) |

---
