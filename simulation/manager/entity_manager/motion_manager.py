from typing import Callable, Set
import numpy as np
import math

from object.entity.motion import Motion, MotionState, MotionConfig
from utils.math_tool import pos_to_tpd, limit_rad, get_euler_rotate_matrix


class MotionManager:
    def __init__(self):
        # 记录每个 Entity 当前激活的运动函数集合
        self.entity_motions = {}
        self.motion = Motion()

    def set_motion_func_set(self, entity, motion_func_set: set[Callable]):
        if entity is None:
            return
        self.entity_motions[entity] = motion_func_set

    def update(self, dt, t):
        for entity, motion_funcs in self.entity_motions.items():
            # 1. 构造当前状态快照，同时置零 tar
            cur_state = MotionState(
                pos=entity.world_pos.copy(),
                vel=entity.world_vel.copy(),
                tpd=entity.world_tpd.copy(),
                rpy=entity.world_rpy.copy(),
                omg=entity.world_omg.copy()
            )
            cur_state.tar_vel = np.zeros(3)
            cur_state.tar_omg = np.zeros(3)

            print(motion_funcs)

            # 应用所有运动函数（设置 tar 分量）
            for func in motion_funcs:
                func(cur_state, t, dt)  # 如果加上t参运动，需修改此处

            # 物理更新：从当前速度向目标速度平滑过渡
            self._apply_physics(cur_state, dt)

            # 回写到entity
            self._apply_motion_state(entity, cur_state)

            # 5. 更新附属部件 (armors)
            if hasattr(entity, 'armors'):
                self._update_armors(entity)

    @staticmethod
    def _apply_physics(state, dt):
        # 斜角运动不归一化就会变成 Minecraft 了 ...
        tar_speed = np.linalg.norm(state.tar_vel)
        if tar_speed > 0:
            norm_speed = min(tar_speed, MotionConfig.T_STEP)
            state.tar_vel = state.tar_vel / tar_speed * norm_speed

        # 计算速度误差
        vel_error = state.tar_vel - state.vel

        # 控制力 = 比例控制 + 阻尼（摩擦力）
        control_force = state.motor_gain * vel_error
        fric_force = -state.fric_coef * state.vel

        # 总力，限制最大力
        total_force = control_force + fric_force
        force_norm = np.linalg.norm(total_force)
        if force_norm > state.max_force:
            total_force = total_force / force_norm * state.max_force

        # 加速度 更新速度
        state.acc = total_force / state.mass
        state.vel += state.acc * dt
        speed = np.linalg.norm(state.vel)
        if speed > state.max_speed:  # 限制最大速度
            state.vel = state.vel / speed * state.max_speed

        # 更新位置
        state.pos += state.vel * dt

        # 旋转部分（类似逻辑）
        omg_error = state.tar_omg - state.omg
        control_torque = state.motor_gain * omg_error
        fric_torque = -state.fric_coef * state.omg

        state.alp = control_torque + fric_torque
        state.omg += state.alp * dt
        rotate_speed = np.linalg.norm(state.omg)
        if rotate_speed > state.max_rotate_speed:  # 限制最大速度
            state.omg = state.omg / rotate_speed * state.max_rotate_speed

        # 更新位姿
        state.rpy += state.omg * dt

        # 角度限制
        state.rpy[2] = limit_rad(state.rpy[2])
        state.rpy[1] = np.clip(state.rpy[1], -np.pi / 2, np.pi / 2)

    @staticmethod
    def _apply_motion_state(entity, state):
        entity.world_pos = state.pos
        entity.world_vel = state.vel
        entity.world_rpy = state.rpy
        entity.world_omg = state.omg
        entity.world_tpd = state.tpd

    def _update_armors(self, robot):
        """
        利用相对坐标更新装甲板，无需在 manager 中维护 init_state。
        """
        # 确保 robot 知道 armor 相对于中心的偏移量
        if not hasattr(robot, '_armors_initialized'):
            for armor in robot.armors:
                # 计算 vector 差
                armor._relative_pos_to_body = armor.world_pos - robot.world_pos
            robot._armors_initialized = True

        rotate_mat = get_euler_rotate_matrix(robot.world_rpy)

        for armor in robot.armors:
            # 核心逻辑：World = RobotWorld + Rot @ LocalOffset
            # 取出预先计算好的相对偏移 (monkey-patched 或原生属性)
            local_pos = getattr(armor, '_relative_pos_to_body', np.zeros(3))

            rotate_offs = rotate_mat @ local_pos

            armor.world_pos = robot.world_pos + rotate_offs
            # v_armor = v_body + w_body x r
            armor.world_vel = robot.world_vel + np.cross(robot.world_omg, rotate_offs)

            # 姿态更新
            armor.world_rpy = robot.world_rpy.copy()
            # 加上装甲板自身的安装角度 (假设均匀分布)
            angle_offset = (armor.armor_id * 2 * math.pi / robot.armor_count)
            armor.world_rpy[2] = limit_rad(robot.world_rpy[2] + math.pi + angle_offset)

            armor.world_omg = robot.world_omg.copy()
            armor.world_tpd = pos_to_tpd(armor.world_pos)

    def instant_stop(self, entity):
        # 强行停止选中的实体
        if entity:
            entity.world_vel[:] = 0
            entity.world_omg[:] = 0
            self.set_motion_func_set(entity, set())




