import numpy as np

from simulation.event_bus import event_bus
from simulation.dataflow import Prediction, DrawText, Tracking
from models.demo4.multi_target_tracker import MultiTargetTracker
from models.demo4.reference_generator import ReferenceGenerator
from models.demo4.mpc_controller import MPCController
from models.demo4.shoot_decider import ShootDecider
from models.demo4.armor_selector import ArmorSelector

from core.algorithms.math import limit_rad


SPIN_THRESH = 3.0                # 装甲板选择器：自旋阈值
LOCK_BIAS = 0.2                   # 装甲板选择器：锁定偏置

V0 = 10.0                         # 弹道解算：初速度
FIRE_THRESHOLD = 0.2             # 射击决策：开火阈值
COOLDOWN = 0.1                    # 射击决策：冷却时间
FIRE_DELAY = 0.1                  # 射击决策：发射延迟

DT = 0.01                         # 参考生成与MPC控制器：时间步长
N = 30                            # 参考生成与MPC控制器：预测步数
YAW_OFFSET = 0.0                  # 参考生成：偏航偏移量
Q_THETA = 100.0                   # MPC：角度权重
Q_DTHETA = 10.0                   # MPC：角速度权重
R_ALPHA = 0.01                    # MPC：控制量权重
ALPHA_MAX = 35                    # MPC：最大角加速度 (deg/s^2)
THETA_MIN = -1e5                  # MPC：角度下限（无硬约束）
THETA_MAX = 1e5                   # MPC：角度上限（无硬约束）


class DemoTracker4:
    def __init__(self, robot_manager):
        self.robot_manager = robot_manager
        self.total_delay = 0.
        self.multi_tracker = MultiTargetTracker()

        # 创建共享的装甲板选择器
        self.armor_selector = ArmorSelector(spin_thresh=SPIN_THRESH, lock_bias=LOCK_BIAS)

        self.shoot_decider = ShootDecider(
            robot_manager,
            self.armor_selector,
            v0=V0,
            fire_threshold=FIRE_THRESHOLD,
            cooldown=COOLDOWN,
            fire_delay=FIRE_DELAY
        )
        self.ref_gen = ReferenceGenerator(
            robot_manager,
            self.armor_selector,
            dt=DT,
            N=N,
            yaw_offset=YAW_OFFSET
        )
        self.mpc_yaw = MPCController(
            dt=DT,
            N=N,
            q_theta=Q_THETA,
            q_dtheta=Q_DTHETA,
            r_alpha=R_ALPHA,
            alpha_max=ALPHA_MAX,
            theta_min=THETA_MIN,
            theta_max=THETA_MAX
        )

        self.obs_armors = None
        self.timestamp = 0.
        event_bus.subscribe('track', self._on_track)

    def _on_track(self, data: Tracking):
        self.obs_armors = data.obs_armors
        self.timestamp = data.timestamp

    def track(self):
        self.multi_tracker.push_observation(self.obs_armors, self.timestamp)
        best_target = self.multi_tracker.get_best_target()

        # 获取自车枪口位置和云台位置
        robot = self.robot_manager.viewing_robot
        if robot is not None:
            gun_rel_pos = robot.get_muzzle_rel_pos()
            gimbal_rel_pos = robot.get_gimbal_rel_pos()
        else:
            gun_rel_pos = np.zeros(3)
            gimbal_rel_pos = np.zeros(3)

        # 发布预测（传入枪口位置）
        self._publish_prediction(best_target, gun_rel_pos, self.timestamp)

        if best_target is None:
            self._publish_plot_data(self.timestamp, None, None, None, None, None, None, None, None)
            return

        # 射击决策
        is_fire = self.shoot_decider.update(best_target, self.timestamp)
        self._publish_fire_command(is_fire)
        pitch_des = self.shoot_decider.latest_pitch
        self.total_delay = self.shoot_decider.total_delay

        # 参考轨迹生成
        theta_ref, omega_ref = self.ref_gen.generate(best_target, self.timestamp, self.total_delay)

        gimbal = robot.gimbal
        gimbal_yaw = gimbal.world_rpy[2]
        gimbal_omg = gimbal.world_omg[2]

        # 调整参考角度（保持不变）
        if len(theta_ref) > 0:
            diff = theta_ref[0] - gimbal_yaw
            k = np.round(diff / (2 * np.pi)).astype(int)
            adjusted = theta_ref[0] - 2 * np.pi * k
            if abs(adjusted - gimbal_yaw) > np.pi:
                k += 1 if adjusted > gimbal_yaw else -1
            theta_ref = theta_ref - 2 * np.pi * k

        x0_yaw = np.array([gimbal_yaw, gimbal_omg])
        theta_des, omega_des, alpha_yaw = self.mpc_yaw.solve(x0=x0_yaw, theta_ref=theta_ref, omega_ref=omega_ref)

        # 计算观测角度
        obs_yaw, obs_pitch = self._calc_obs_angle()

        # 计算预测角度（传入云台位置和枪口位置）
        pred_yaw, pred_pitch = self._calc_pred_angle(best_target, gimbal_rel_pos, gun_rel_pos)

        # 设置云台目标
        gimbal.set_target(theta_des, omega_des, alpha_yaw, pitch_des, 0., 0.)

        # 发布绘图数据
        self._publish_plot_data(self.timestamp, obs_yaw, obs_pitch, pred_yaw, pred_pitch,
                                limit_rad(theta_des), limit_rad(gimbal.world_rpy[2]), gimbal.world_rpy[1], pitch_des)

        self._publish_draw_text(f'alpha_yaw: {alpha_yaw:.2f}')

    def _calc_obs_angle(self):
        """计算第一个观测装甲板相对于云台的yaw和pitch"""
        if not self.obs_armors:
            return None, None
        obs = self.obs_armors[0]
        gimbal_rel_pos = self.robot_manager.viewing_robot.get_gimbal_rel_pos()
        dx = obs.rel_pos[0] - gimbal_rel_pos[0]
        dy = obs.rel_pos[1] - gimbal_rel_pos[1]
        dz = obs.rel_pos[2] - gimbal_rel_pos[2]
        yaw = np.arctan2(dy, dx)
        pitch = -np.arctan2(dz, np.sqrt(dx * dx + dy * dy))
        return yaw, pitch

    def _calc_pred_angle(self, target, gimbal_rel_pos, gun_rel_pos):
        pred_center, pred_armor = target.get_pred_pos(self.total_delay, gun_rel_pos, self.armor_selector)
        if pred_armor is None:
            return None, None
        dx = pred_armor[0] - gimbal_rel_pos[0]
        dy = pred_armor[1] - gimbal_rel_pos[1]
        dz = pred_armor[2] - gimbal_rel_pos[2]
        yaw = np.arctan2(dy, dx)
        pitch = -np.arctan2(dz, np.sqrt(dx * dx + dy * dy))
        return yaw, pitch

    def _publish_plot_data(self, timestamp, obs_yaw, obs_pitch, pred_yaw, pred_pitch,
                           mpc_yaw, actual_yaw, actual_pitch, ballistic_pitch):
        """发布绘图数据事件"""
        event_bus.publish('plot', {
            'timestamp': timestamp,
            'obs_yaw': obs_yaw,
            'obs_pitch': obs_pitch,
            'pred_yaw': pred_yaw,
            'pred_pitch': pred_pitch,
            'mpc_yaw': mpc_yaw,
            'actual_yaw': actual_yaw,
            'actual_pitch': actual_pitch,
            'ballistic_pitch': ballistic_pitch
        })

    def _publish_prediction(self, target, gun_rel_pos, timestamp):
        if target is None:
            pred = Prediction(
                center=np.array([]),
                armors=[],
                timestamp=timestamp,
                is_tracking=False,
                fps=0.0,
                state_vector=None
            )
        else:
            pred_center, pred_armor = target.get_pred_pos(self.total_delay, gun_rel_pos, self.armor_selector)
            pred = Prediction(
                center=pred_center,
                armors=[pred_armor] if pred_armor is not None else [],
                timestamp=timestamp,
                is_tracking=True,
                fps=0.0,
                state_vector=target.ekf.ekf.x
            )
        event_bus.publish('pred', pred)

    def _publish_fire_command(self, is_fire):
        """发布开火指令"""
        event_bus.publish('fire', {'is_fire': is_fire})

    def _publish_draw_text(self, text, color=(255, 255, 255)):
        """发布绘图文本"""
        event_bus.publish('draw', DrawText(text, color))


