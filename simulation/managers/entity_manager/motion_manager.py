from typing import Callable
import numpy as np

from core.entities.rigid.chassis import Chassis
from core.entities.rigid.gimbal import Gimbal
from core.entities.property.motion import Motion, MotionState, MotionConfig
from core.algorithms.math import limit_rad


class MotionManager:
    def __init__(self):
        # 分别存储不同实体的运动函数集
        self.entity_motions = {}  # 仍使用字典，键为实体对象
        self.chassis_list = []
        self.gimbal_list = []

    def add_entity(self, entity):
        """将实体添加到管理器中，并分类"""
        self.entity_motions[entity] = set()
        if isinstance(entity, Chassis):
            self.chassis_list.append(entity)
        elif isinstance(entity, Gimbal):
            self.gimbal_list.append(entity)

    def remove_entity(self, entity):
        if entity in self.entity_motions:
            del self.entity_motions[entity]
        if entity in self.chassis_list:
            self.chassis_list.remove(entity)
        if entity in self.gimbal_list:
            self.gimbal_list.remove(entity)

    def set_motion_func_set(self, entity, motion_func_set: set[Callable]):
        if entity in self.entity_motions:
            self.entity_motions[entity] = motion_func_set

    def update(self, dt, t):
        # 1. 更新底盘
        for chassis in self.chassis_list:
            self._update_entity(chassis, dt, t)
            chassis.update_armors()

        # 2. 更新云台
        for gimbal in self.gimbal_list:
            if gimbal.auto_aiming:
                # 自动瞄准模式下，用闭环控制
                gimbal.apply_control(dt)
                # 更新位置依赖底盘
                if gimbal.owner_chassis is not None:
                    gimbal.update_from_chassis(gimbal.owner_chassis)
            else:
                # 手动模式：物理更新（键盘控制）
                self._update_entity(gimbal, dt, t)
                if gimbal.owner_chassis is not None:
                    gimbal.update_from_chassis(gimbal.owner_chassis)

            gimbal.update_children()

    def _update_entity(self, entity, dt, t):
        """通用物理更新，使用 MotionState"""
        motion_funcs = self.entity_motions.get(entity, set())
        # 构造当前状态快照
        cur_state = MotionState(
            pos=entity.world_pos.copy(),
            vel=entity.world_vel.copy(),
            tpd=entity.world_tpd.copy(),
            rpy=entity.world_rpy.copy(),
            omg=entity.world_omg.copy(),
        )
        cur_state.tar_vel = np.zeros(3)
        cur_state.tar_omg = np.zeros(3)

        # 应用所有运动函数
        for func in motion_funcs:
            func(cur_state, t, dt)

        # 物理更新
        self._apply_physics_trans(cur_state, dt)
        self._apply_physics_rot(cur_state, dt)

        # 回写
        entity.world_pos = cur_state.pos
        entity.world_vel = cur_state.vel
        entity.world_rpy = cur_state.rpy
        entity.world_omg = cur_state.omg
        entity.world_tpd = cur_state.tpd

    @staticmethod
    def _apply_physics_trans(state, dt):
        tar_speed = np.linalg.norm(state.tar_vel)
        if tar_speed > 0:
            norm_speed = min(tar_speed, MotionConfig.T_STEP)
            state.tar_vel = state.tar_vel / tar_speed * norm_speed

        vel_error = state.tar_vel - state.vel
        control_force = state.motor_gain * vel_error
        fric_force = -state.fric_coef * state.vel
        total_force = control_force + fric_force
        force_norm = np.linalg.norm(total_force)
        if force_norm > state.max_force:
            total_force = total_force / force_norm * state.max_force

        state.acc = total_force / state.mass
        state.vel += state.acc * dt
        speed = np.linalg.norm(state.vel)
        if speed > state.max_speed:
            state.vel = state.vel / speed * state.max_speed

        state.pos += state.vel * dt

    @staticmethod
    def _apply_physics_rot(state, dt):
        omg_error = state.tar_omg - state.omg
        control_torque = state.motor_gain * omg_error
        fric_torque = -state.fric_coef * state.omg
        state.alp = control_torque + fric_torque
        state.omg += state.alp * dt
        rotate_speed = np.linalg.norm(state.omg)
        if rotate_speed > state.max_rotate_speed:
            state.omg = state.omg / rotate_speed * state.max_rotate_speed

        state.rpy += state.omg * dt
        state.rpy[1] = np.clip(state.rpy[1], -np.pi / 2, np.pi / 2)

    def instant_stop(self, entity):
        if entity:
            entity.world_vel[:] = 0
            entity.world_omg[:] = 0
            self.set_motion_func_set(entity, set())