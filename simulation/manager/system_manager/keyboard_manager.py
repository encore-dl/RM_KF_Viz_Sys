import pygame as pg
from object.entity.robot import RobotType


class KeyboardManager:
    def __init__(self, simulator):
        self.simulator = simulator
        self.motion = simulator.motion_manager.motion

        self.pressed_keys = set()

        # 映射
        self.trans_key_func_map = {
            pg.K_UP: self.motion.go_up,
            pg.K_DOWN: self.motion.go_down,
            pg.K_LEFT: self.motion.go_left,
            pg.K_RIGHT: self.motion.go_right,
            pg.K_g: self.motion.ascend,
            pg.K_b: self.motion.descend,
        }

        self.rot_key_func_map = {
            pg.K_a: self.motion.rotate_anticlockwise,
            pg.K_d: self.motion.rotate_clockwise,
            pg.K_z: self.motion.top_rotate_anticlockwise,
            pg.K_c: self.motion.top_rotate_clockwise,
            pg.K_w: self.motion.pitch_up,
            pg.K_x: self.motion.pitch_down,
        }

        # 功能键映射
        self.spec_key_func_map = {
            pg.K_ESCAPE: lambda: 'escape',
            pg.K_r: lambda: 'reset',
            pg.K_1: lambda: self.simulator.select_entity('robot', 0),
            pg.K_2: lambda: self.simulator.select_entity('robot', 1),  # 示例
            pg.K_9: lambda: self.simulator.select_entity('camera'),
            pg.K_KP9: self.simulator.camera_manager.switch_auto_aiming,
            pg.K_SPACE: lambda: self.simulator.motion_manager.instant_stop(self.simulator.selected_entity)
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
        # 前面处理combo键
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
        # 如果没有触发任何 Combo，再处理单键逻辑
        if key in self.spec_key_func_map:
            return self.spec_key_func_map[key]()
        return None

    def handle_motion_key(self):
        # 对在 pressed_keys 中的 motion key 进行统一处理

        entity = self.simulator.selected_entity
        if entity is None:
            return

        curr_motions = set()

        # 处理平移和旋转
        has_trans = False
        for key, func in self.trans_key_func_map.items():
            if key in self.pressed_keys:
                curr_motions.add(func)
                has_trans = True

        has_rot = False
        for key, func in self.rot_key_func_map.items():
            if key in self.pressed_keys:
                curr_motions.add(func)
                has_rot = True

        # 3. 将构建好的集合应用给选中的实体
        self.simulator.motion_manager.set_motion_func_set(entity, curr_motions)





