import pygame as pg
from core.entities.property.robot_type import RobotType
from core.entities.rigid.robot import Robot
from core.entities.property.motion import Motion


class KeyboardManager:
    def __init__(self, simulator):
        self.simulator = simulator
        self.motion = Motion()
        self.pressed_keys = set()

        # 平移键（影响底盘）
        self.trans_keys = {
            pg.K_UP: self.motion.go_up,
            pg.K_DOWN: self.motion.go_down,
            pg.K_LEFT: self.motion.go_left,
            pg.K_RIGHT: self.motion.go_right,
            pg.K_g: self.motion.ascend,
            pg.K_b: self.motion.descend,
        }

        # 底盘旋转键
        self.chassis_rot_keys = {
            pg.K_a: self.motion.rotate_chassis_anticlockwise,
            pg.K_d: self.motion.rotate_chassis_clockwise,
            pg.K_z: self.motion.rotate_chassis_fast_anticlockwise,
            pg.K_c: self.motion.rotate_chassis_fast_clockwise,
        }

        # 云台旋转键（相对）
        self.gimbal_rot_keys = {
            pg.K_q: self.motion.rotate_gimbal_yaw_left,
            pg.K_e: self.motion.rotate_gimbal_yaw_right,
            pg.K_w: self.motion.rotate_gimbal_pitch_up,
            pg.K_x: self.motion.rotate_gimbal_pitch_down,
        }

        # 特殊功能键
        self.spec_key_func_map = {
            pg.K_ESCAPE: lambda: 'escape',
            pg.K_r: lambda: 'reset',
            pg.K_SPACE: lambda: self.simulator.motion_manager.instant_stop(self.simulator.robot_manager.controlled_robot.chassis),
            pg.K_KP9: lambda: self.simulator.robot_manager.viewing_robot.gimbal.switch_auto_aiming() if self.simulator.robot_manager.viewing_robot else None,
        }

    def handle_event(self, event):
        result = None
        if event.type == pg.KEYDOWN:
            self.pressed_keys.add(event.key)
            combo_result = self.handle_combo_key(event.key)
            if combo_result:
                result = combo_result
            else:
                result = self.handle_single_key(event.key)
        elif event.type == pg.KEYUP:
            if event.key in self.pressed_keys:
                self.pressed_keys.remove(event.key)
        self.handle_motion_key()
        return result

    def handle_combo_key(self, key):
        if pg.K_BACKSPACE in self.pressed_keys:
            if pg.K_0 <= key <= pg.K_9:
                robot_id = key - pg.K_0
                self.simulator.robot_manager.delete_robot(robot_id)
                return 'combo'

        if pg.K_RETURN in self.pressed_keys:
            if key == pg.K_1:
                self.simulator.robot_manager.create_robot(RobotType.Hero)
                return 'combo'
            elif key == pg.K_2:
                self.simulator.robot_manager.create_robot(RobotType.Sentry)
                return 'combo'
        return None

    def handle_single_key(self, key):
        if key == pg.K_1:
            self.simulator.robot_manager.switch_control_robot()
            return 'switch_control'
        elif key == pg.K_9:
            self.simulator.robot_manager.switch_view_robot()
            return 'switch_view'
        elif key in self.spec_key_func_map and self.spec_key_func_map[key] is not None:
            return self.spec_key_func_map[key]()
        return None

    def handle_motion_key(self):
        entity = self.simulator.robot_manager.controlled_robot
        if not isinstance(entity, Robot):
            return

        chassis = entity.chassis
        gimbal = entity.gimbal

        # 底盘运动函数始终设置
        chassis_motions = set()
        for key, func in self.trans_keys.items():
            if key in self.pressed_keys:
                chassis_motions.add(func)
        for key, func in self.chassis_rot_keys.items():
            if key in self.pressed_keys:
                chassis_motions.add(func)
        self.simulator.motion_manager.set_motion_func_set(chassis, chassis_motions)

        # 云台运动函数仅在非自动瞄准时设置
        if not gimbal.auto_aiming:
            gimbal_motions = set()
            for key, func in self.gimbal_rot_keys.items():
                if key in self.pressed_keys:
                    gimbal_motions.add(func)
            self.simulator.motion_manager.set_motion_func_set(gimbal, gimbal_motions)
        else:
            # 自动瞄准时清空云台的运动函数
            self.simulator.motion_manager.set_motion_func_set(gimbal, set())