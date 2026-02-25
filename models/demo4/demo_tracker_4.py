import numpy as np
import time
from simulation.event_bus import event_bus
from simulation.dataflow import Prediction, DrawText
from models.demo4.multi_target_tracker import MultiTargetTracker
from models.demo4.reference_generator import ReferenceGenerator
from models.demo4.mpc_controller import MPCController
from simulation.managers.system_manager.shoot_decider import ShootDecider  # 新增

class DemoTracker4:
    def __init__(self, camera_manager, fly_time=0.4):
        self.camera_manager = camera_manager
        self.fly_time = fly_time
        self.multi_tracker = MultiTargetTracker(camera_manager=camera_manager)
        self.ref_gen = ReferenceGenerator(camera_manager)
        self.mpc = MPCController(
            dt=0.01, N=20,
            theta_min=-1e9, theta_max=1e9
        )
        # 初始化射击决策器
        self.shoot_decider = ShootDecider(camera_manager, v0=30.0, fire_threshold=0.05, cooldown=0.1)

    def track(self, obs_armors, dt, t_stamp):
        self.multi_tracker.push_observation(obs_armors, t_stamp)
        best_target = self.multi_tracker.get_best_target()

        if best_target is None:
            self._publish_prediction(None, t_stamp)
            alpha = -10.0 * self.camera_manager.selected_camera.world_omg[2] if abs(self.camera_manager.selected_camera.world_omg[2]) >= 0.01 else 0.0
            event_bus.publish('gimbal', {'alpha': alpha, 'timestamp': t_stamp})
            return

        # 生成参考轨迹（用于控制）
        theta_ref, omega_ref, phi_ref = self.ref_gen.generate(best_target.ekf, t_stamp)
        gimbal_yaw = self.camera_manager.selected_camera.world_rpy[2]
        gimbal_omg = self.camera_manager.selected_camera.world_omg[2]

        # 相位对齐
        if len(theta_ref) > 0:
            diff = theta_ref[0] - gimbal_yaw
            # 计算使 theta_ref 调整后与 gimbal_yaw 最接近的圈数
            k = np.round(diff / (2 * np.pi)).astype(int)
            # 但需确保调整后误差在 (-π, π] 内
            adjusted = theta_ref[0] - 2 * np.pi * k
            if abs(adjusted - gimbal_yaw) > np.pi:
                k += 1 if adjusted > gimbal_yaw else -1
            theta_ref = theta_ref - 2 * np.pi * k

        mpc_x0 = np.array([gimbal_yaw, gimbal_omg])
        alpha = self.mpc.solve(x0=mpc_x0, theta_ref=theta_ref, omega_ref=omega_ref)

        event_bus.publish('draw', DrawText(f'alpha: {alpha:.2f}', (255, 255, 255)))
        event_bus.publish('gimbal', {'alpha': alpha, 'timestamp': t_stamp})

        # 射击决策
        self.shoot_decider.update(best_target.ekf, t_stamp)

        # 发布预测可视化（使用预设飞行时间）
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