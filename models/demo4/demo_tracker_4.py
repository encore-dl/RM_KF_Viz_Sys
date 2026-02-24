import numpy as np
import time
from simulation.event_bus import event_bus
from simulation.dataflow import Prediction, DrawText
from models.demo4.multi_target_tracker import MultiTargetTracker
from models.demo4.reference_generator import ReferenceGenerator
from models.demo4.mpc_controller import MPCController
from simulation.managers.system_manager.shoot_decider import ShootDecider  # 新增

class DemoTracker4:
    def __init__(self, camera, fly_time=0.4):
        self.camera = camera
        self.fly_time = fly_time
        self.multi_tracker = MultiTargetTracker(camera=camera)
        self.ref_gen = ReferenceGenerator(camera)
        self.mpc = MPCController(
            dt=0.01, N=20,
            theta_min=-1e9, theta_max=1e9
        )
        # 初始化射击决策器
        self.shoot_decider = ShootDecider(camera, v0=30.0, fire_threshold=0.05, cooldown=0.1)

    def track(self, obs_armors, dt, t_stamp):
        self.multi_tracker.push_observation(obs_armors, t_stamp)
        best_target = self.multi_tracker.get_best_target()

        if best_target is None:
            self._publish_prediction(None, t_stamp)
            alpha = -10.0 * self.camera.world_omg[2] if abs(self.camera.world_omg[2]) >= 0.01 else 0.0
            event_bus.publish('gimbal_command', {'alpha': alpha, 'timestamp': t_stamp})
            return

        # 生成参考轨迹（用于控制）
        theta_ref, omega_ref, phi_ref = self.ref_gen.generate(best_target.ekf, t_stamp)
        gimbal_yaw = self.camera.world_rpy[2]
        gimbal_omg = self.camera.world_omg[2]

        # 相位对齐
        if len(theta_ref) > 0:
            # 计算第一个点与当前角度的差，并四舍五入到最近的2π倍数
            diff = theta_ref[0] - gimbal_yaw
            k = int(round(diff / (2 * np.pi)))  # 找到合适的圈数
            theta_ref = theta_ref - 2 * np.pi * k  # 调整整个轨迹

        mpc_x0 = np.array([gimbal_yaw, gimbal_omg])
        alpha = self.mpc.solve(x0=mpc_x0, theta_ref=theta_ref, omega_ref=omega_ref)

        event_bus.publish('draw', DrawText(f'alpha: {alpha:.2f}', (255, 255, 255)))
        event_bus.publish('gimbal_command', {'alpha': alpha, 'timestamp': t_stamp})

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