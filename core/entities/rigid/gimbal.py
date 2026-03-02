import numpy as np
from core.entities.rigid.rigid import Rigid
from core.entities.rigid.camera import Camera
from core.entities.rigid.muzzle import Muzzle
from core.algorithms.math import euler_to_rotation_matrix, limit_rad
from core.algorithms.control.pid import PIDParams, CascadedPID


class Gimbal(Rigid):
    """云台类，包含相机和枪口，安装在底盘上，具有相对旋转"""
    def __init__(self, mount_pos=np.zeros(3), mount_rpy=np.zeros(3), **kwargs):
        super().__init__(**kwargs)
        self.mount_pos = mount_pos.copy()
        self.mount_rpy = mount_rpy.copy()
        self.owner_chassis = None

        self.camera = Camera()
        self.muzzle = Muzzle()

        self.auto_aiming = False
        self.target_theta = 0.0      # 期望偏航角（世界）
        self.target_omega = 0.0       # 期望偏航角速度（前馈）
        self.target_alpha = 0.0       # 期望偏航角加速度（前馈）
        self.target_pitch = 0.0       # 期望俯仰角
        self.target_pitch_omega = 0.0 # 期望俯仰角速度（前馈）
        self.target_pitch_alpha = 0.0 # 期望俯仰角加速度（前馈）

        # 动力学参数（可调）
        self.inertia_yaw = 0.01          # 偏航转动惯量 (kg*m^2)
        self.inertia_pitch = 0.01        # 俯仰转动惯量
        self.damping_yaw = 0.001         # 偏航阻尼系数
        self.damping_pitch = 0.001       # 俯仰阻尼系数
        self.motor_torque_max_yaw = 1.0  # 偏航最大扭矩 (Nm)
        self.motor_torque_max_pitch = 1.0  # 俯仰最大扭矩

        self.yaw_pid = CascadedPID(
            outer_params=PIDParams(kp=7, ki=0.0, kd=0.,
                                   max_output=50.0, max_integral=0.0),
            inner_params=PIDParams(kp=25.0, ki=0.0, kd=0.0,
                                   max_output=80.0, max_integral=0.0)
        )

        # 俯仰轴（与偏航相同，但可根据实际情况微调）
        self.pitch_pid = CascadedPID(
            outer_params=PIDParams(kp=20, ki=0.0, kd=0.1,
                                   max_output=20.0, max_integral=0.0),
            inner_params=PIDParams(kp=40.0, ki=1.0, kd=0.0,
                                   max_output=80.0, max_integral=0.0)
        )

        self.update_children()

    def set_target(self, theta, omega, alpha, pitch, pitch_omega, pitch_alpha):
        """设置期望状态"""
        self.target_theta = theta
        self.target_omega = omega
        self.target_alpha = alpha
        self.target_pitch = pitch
        self.target_pitch_omega = pitch_omega
        self.target_pitch_alpha = pitch_alpha

    def set_mount_pose(self, pos, rpy):
        self.mount_pos = pos.copy()
        self.mount_rpy = rpy.copy()

    def update_from_chassis(self, chassis):
        if chassis is None:
            return
        R_chassis = euler_to_rotation_matrix(chassis.world_rpy)

        self.world_pos = chassis.world_pos + R_chassis @ self.mount_pos

        r_vec = self.world_pos - chassis.world_pos
        self.world_vel = chassis.world_vel + np.cross(chassis.world_omg, r_vec)

    def update_children(self):
        R_gimbal = euler_to_rotation_matrix(self.world_rpy)

        self.camera.world_pos = self.world_pos + R_gimbal @ self.camera.mount_pos
        self.camera.world_rpy = self.world_rpy + self.camera.mount_rpy
        self.camera.world_vel = self.world_vel + np.cross(self.world_omg, self.camera.world_pos - self.world_pos)
        self.camera.world_omg = self.world_omg.copy()

        self.muzzle.world_pos = self.world_pos + R_gimbal @ self.muzzle.mount_pos
        self.muzzle.world_rpy = self.world_rpy + self.muzzle.mount_rpy
        self.muzzle.world_vel = self.world_vel + np.cross(self.world_omg, self.muzzle.world_pos - self.world_pos)
        self.muzzle.world_omg = self.world_omg.copy()

    def apply_control(self, dt):
        if not self.auto_aiming:
            return

        # ----- 偏航轴控制（带角度Wrap）-----
        # 计算误差并调整期望角度，使误差最小
        error_theta = limit_rad(self.target_theta - self.world_rpy[2])
        setpoint_theta_adj = self.world_rpy[2] + error_theta  # 调整后的期望角度（与当前角度最近）

        alpha_cmd = self.yaw_pid.update(
            setpoint=setpoint_theta_adj,
            measurement_outer=self.world_rpy[2],
            measurement_inner=self.world_omg[2],
            feedforward_outer=self.target_omega,
            feedforward_inner=self.target_alpha,
            dt=dt
        )
        alpha_cmd = np.clip(alpha_cmd, -50.0, 50.0)
        self.world_alp[2] = alpha_cmd
        self.world_omg[2] += self.world_alp[2] * dt
        self.world_rpy[2] += self.world_omg[2] * dt

        # ----- 俯仰轴控制（俯仰范围有限，通常不需Wrap，但也可做简单处理）-----
        # 俯仰角限制在[-π/2, π/2]，一般不跨越边界，直接使用目标值即可
        pitch_alpha_cmd = self.pitch_pid.update(
            setpoint=self.target_pitch,
            measurement_outer=self.world_rpy[1],
            measurement_inner=self.world_omg[1],
            feedforward_outer=self.target_pitch_omega,
            feedforward_inner=self.target_pitch_alpha,
            dt=dt
        )
        pitch_alpha_cmd = np.clip(pitch_alpha_cmd, -50.0, 50.0)
        self.world_alp[1] = pitch_alpha_cmd
        self.world_omg[1] += self.world_alp[1] * dt
        self.world_rpy[1] += self.world_omg[1] * dt
        self.world_rpy[1] = np.clip(self.world_rpy[1], -np.pi / 2, np.pi / 2)

    def switch_auto_aiming(self):
        self.auto_aiming = not self.auto_aiming


