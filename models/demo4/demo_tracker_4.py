import numpy as np
import time
from simulation.event_bus import event_bus
from simulation.dataflow import Prediction, DrawText
from models.demo4.multi_target_tracker import MultiTargetTracker
from models.demo4.reference_generator import ReferenceGenerator
from models.demo4.mpc_controller import MPCController
from simulation.managers.system_manager.shoot_decider import ShootDecider


class DemoTracker4:
    def __init__(self, robot_manager, bullet_manager, fly_time=0.4):
        self.robot_manager = robot_manager
        self.bullet_manager = bullet_manager
        self.fly_time = fly_time
        self.multi_tracker = MultiTargetTracker(robot_manager=robot_manager)
        self.ref_gen = ReferenceGenerator(robot_manager)
        self.mpc_yaw = MPCController(
            dt=0.01, N=20,
            q_theta=100., q_dtheta=10., r_alpha=0.01, alpha_max=35,
            theta_min=-1e9, theta_max=1e9
        )
        self.mpc_pitch = MPCController(
            dt=0.01, N=20,
            q_theta=100., q_dtheta=10., r_alpha=0.01, alpha_max=35,
            theta_min=-np.pi / 2, theta_max=np.pi / 2
        )
        self.shoot_decider = ShootDecider(robot_manager, bullet_manager, v0=30.0, fire_threshold=0.05, cooldown=0.1)

    def track(self, obs_armors, dt, t_stamp):
        self.multi_tracker.push_observation(obs_armors, t_stamp)
        best_target = self.multi_tracker.get_best_target()

        if best_target is None:
            self._publish_prediction(None, t_stamp)
            event_bus.publish('gimbal_yaw', {'alpha': 0.0, 'timestamp': t_stamp})
            event_bus.publish('gimbal_pitch', {'alpha': 0.0, 'timestamp': t_stamp})
            return

        self.shoot_decider.update(best_target.ekf, t_stamp)
        latest_pitch = self.shoot_decider.latest_pitch

        theta_ref, omega_ref, phi_ref, phi_omega_ref = self.ref_gen.generate(best_target.ekf, t_stamp)
        # if abs(latest_pitch) > 1e-6:
        #     phi_ref[:] = latest_pitch
        #     phi_omega_ref[:] = 0.0  # 假设期望 pitch 恒定，角速度为 0

        camera = self.robot_manager.selected_robot.get_camera()
        gimbal_yaw = camera.world_rpy[2]
        gimbal_omg = camera.world_omg[2]
        gimbal_pitch = camera.world_rpy[1]
        gimbal_pitch_omg = camera.world_omg[1]

        # 相位对齐（yaw 解缠）
        if len(theta_ref) > 0:
            diff = theta_ref[0] - gimbal_yaw
            k = np.round(diff / (2 * np.pi)).astype(int)
            adjusted = theta_ref[0] - 2 * np.pi * k
            if abs(adjusted - gimbal_yaw) > np.pi:
                k += 1 if adjusted > gimbal_yaw else -1
            theta_ref = theta_ref - 2 * np.pi * k

        x0_yaw = np.array([gimbal_yaw, gimbal_omg])
        theta_des, omega_des, alpha_yaw = self.mpc_yaw.solve(x0=x0_yaw, theta_ref=theta_ref, omega_ref=omega_ref)

        x0_pitch = np.array([gimbal_pitch, gimbal_pitch_omg])
        pitch_des, pitch_omega_des, alpha_pitch = self.mpc_pitch.solve(x0=x0_pitch, theta_ref=phi_ref, omega_ref=phi_omega_ref)

        # 设置云台期望值
        gimbal = self.robot_manager.selected_robot.gimbal
        gimbal.set_target(theta_des, omega_des, alpha_yaw, pitch_des, pitch_omega_des, alpha_pitch)

        event_bus.publish('draw', DrawText(f'alpha_yaw: {alpha_yaw:.2f}', (255, 255, 255)))
        event_bus.publish('draw', DrawText(f'alpha_pitch: {alpha_pitch:.2f}', (255, 255, 255)))

        pred_center, pred_armor = best_target.ekf.get_pred_pos(self.fly_time)
        pred = Prediction(
            center=pred_center,
            armors=[pred_armor] if pred_armor is not None else [],
            timestamp=time.time(),
            is_tracking=True,
            fps=0.0,
            state_vector=best_target.ekf.ekf.x
        )
        event_bus.publish('pred', pred)

    def _publish_prediction(self, target, t_stamp):
        pred = Prediction(
            center=np.array([]),
            armors=[],
            timestamp=time.time(),
            is_tracking=False,
            fps=0.0,
            state_vector=None
        )
        event_bus.publish('pred', pred)