import numpy as np

from core.algorithms.trajectory.trajectory_solver import TrajectorySolver
from simulation.dataflow.draw import DrawText
from simulation.event_bus import event_bus
from models.demo4.armor_selector import ArmorSelector
from core.algorithms.math.angle import limit_rad


class ShootDecider:
    def __init__(self, robot_manager, selector: ArmorSelector, v0=10.0, fire_threshold=0.02, cooldown=0.1, fire_delay=0.1):
        self.robot_manager = robot_manager
        self.selector = selector
        self.v0 = v0
        self.fire_threshold = fire_threshold
        self.cooldown = cooldown
        self.fire_delay = fire_delay
        self.total_delay = None
        self.last_fire_time = 0
        self.traj_solver = TrajectorySolver(k=0.002043, N=200)
        self.latest_pitch = 0.

    def update(self, target, current_time):
        if target is None or not target.ekf.is_init:
            return False

        robot = self.robot_manager.viewing_robot
        if robot is None:
            return False
        gun_rel_pos = robot.get_muzzle_rel_pos()

        # 粗略估计飞行时间
        center_pos = target.ekf.ekf.x[:3]
        vec = center_pos - gun_rel_pos
        dist = np.linalg.norm(vec)
        if dist < 1e-6:
            return False
        fly_time_est = dist / self.v0
        print(f"fly_time_est: {fly_time_est}")

        # 使用装甲板选择器获取未来时刻的最佳装甲板
        armor_id, armor_pos = self.selector.select_armor(target, gun_rel_pos, fly_time_est)

        # 弹道解算
        target_rel = armor_pos - gun_rel_pos
        # 在获得 target_rel 后
        dx, dy, dz = target_rel
        x0 = np.sqrt(dx * dx + dy * dy)
        y0 = dz

        # 计算无阻力低抛角
        g = self.traj_solver.g
        v0 = self.v0
        if x0 < 1e-6:  # 接近垂直情况
            pitch_guess = np.arctan2(y0, x0) if abs(x0) > 0 else 0.0
        else:
            A = g * x0 ** 2 / (2 * v0 ** 2)
            B = x0
            C = y0 + A
            discriminant = B**2 - 4*A*C
            if discriminant < 0:
                return False
            sqrt_disc = np.sqrt(discriminant)
            u_low = (B - sqrt_disc) / (2 * A)  # 低抛对应的 tanθ
            pitch_guess = np.arctan(u_low)

        # 第一次求解
        fly_time, pitch = self.traj_solver.solve(v0, target_rel, pitch_guess)
        if fly_time is None or pitch is None:
            return False

        # 用精确飞行时间重新预测装甲板
        armor_id, armor_pos = self.selector.select_armor(target, gun_rel_pos, fly_time)
        target_rel = armor_pos - gun_rel_pos
        fly_time, pitch = self.traj_solver.solve(v0, target_rel, pitch)  # 以上次 pitch 为初值
        if fly_time is None or pitch is None:
            return False
        # armor_id, armor_pos = self.selector.select_armor(target, gun_rel_pos, fly_time)
        # target_rel = armor_pos - gun_rel_pos
        # fly_time, pitch = self.traj_solver.solve(v0, target_rel, pitch)  # 以上次 pitch 为初值
        # if fly_time is None or pitch is None:
        #     return False

        # self.total_delay = fly_time
        self.total_delay = fly_time + self.fire_delay

        desired_yaw = np.arctan2(target_rel[1], target_rel[0])
        desired_pitch = -pitch
        self.latest_pitch = desired_pitch

        gimbal = robot.gimbal
        current_yaw = gimbal.world_rpy[2]
        current_pitch = gimbal.world_rpy[1]

        yaw_diff = abs(limit_rad(desired_yaw - current_yaw))
        pitch_diff = abs(desired_pitch - current_pitch)

        event_bus.publish('draw', DrawText(f'yaw_diff: {yaw_diff:.3f}', (255,255,255)))
        event_bus.publish('draw', DrawText(f'pitch_diff: {pitch_diff:.3f}', (255,255,255)))

        aim_ok = yaw_diff < self.fire_threshold and pitch_diff < self.fire_threshold
        cooldown_ok = current_time - self.last_fire_time >= self.cooldown
        if aim_ok and cooldown_ok:
            self.last_fire_time = current_time
            return True

        return False



