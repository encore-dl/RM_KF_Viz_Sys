import pygame as pg
from typing import Optional

from object.entity.robot import RobotType


class KeyboardManager:
    COMBO_FIRST_KEY = {
        pg.K_BACKSPACE, pg.K_RETURN
    }

    KEY_TO_NUM_MAP = {
        pg.K_0: 0,
        pg.K_1: 1,
        pg.K_2: 2,
        pg.K_3: 3,
        pg.K_4: 4,
        pg.K_5: 5,
        pg.K_6: 6,
        pg.K_7: 7,
        pg.K_8: 8,
        pg.K_9: 9
    }

    MOTION_KEY_LIST = [
        pg.K_UP, pg.K_DOWN, pg.K_LEFT, pg.K_RIGHT,
        pg.K_a, pg.K_d, pg.K_z, pg.K_c, pg.K_w, pg.K_x,
        pg.K_g, pg.K_b,
        pg.K_KP1, pg.K_KP2, pg.K_KP3, pg.K_KP4, pg.K_KP5, pg.K_KP6
    ]

    def __init__(self):
        self.pressed_keys = set()
        self.combo_mode = False
        self.combo_first_key: Optional[int] = None
        self.is_key_up = False

    def handle_event(self, event, simulator):
        if event.type == pg.KEYDOWN:
            self.pressed_keys.add(event.key)
            self.is_key_up = False

            if event.key in self.COMBO_FIRST_KEY:
                self.combo_mode = True
                self.combo_first_key = event.key
                return

            if self.combo_mode:
                self.do_combo_key(event.key, simulator)
                self.combo_mode = False
                return

            self.do_single_key(event.key, simulator)
        elif event.type == pg.KEYUP:
            self.pressed_keys.remove(event.key)
            self.is_key_up = True

            if self.combo_mode and event.key == self.combo_first_key:  # combo 模式的第一个键
                self.combo_mode = False

    def update(self, simulator):
        # 最后一个键松开后，运动停止
        if self.is_key_up and len(self.pressed_keys) == 0:
            simulator.motion_manager.set_motion(
                entity=simulator.selected_entity,
                motion_func=simulator.motion_manager.motion.stay_still
            )
            self.is_key_up = False
            return
        #
        if self.is_key_up and len(self.pressed_keys) > 0:
            self.do_current_keys(simulator)
            self.is_key_up = False
            return

        if len(self.pressed_keys) > 0:
            self.do_current_keys(simulator)

    def do_combo_key(self, second_key, simulator):
        if self.combo_first_key == pg.K_BACKSPACE:
            robot_num = self.KEY_TO_NUM_MAP.get(second_key)
            if robot_num is not None:
                simulator.robot_manager.delete_robot(robot_num)
        elif self.combo_first_key == pg.K_RETURN:
            if second_key == pg.K_1:
                simulator.robot_manager.create_robot(RobotType.Hero)

    @staticmethod
    def do_single_key(key, simulator):
        if key == pg.K_ESCAPE:
            return 'escape'
        elif key == pg.K_1:
            simulator.select_entity('robot', 0)
        elif key == pg.K_2:
            simulator.select_entity('camera')
        elif key == pg.K_KP9:
            simulator.camera_manager.camera.auto_aiming = not simulator.camera_manager.camera.auto_aiming
        elif key == pg.K_SPACE:
            simulator.motion_manager.set_motion(
                entity=simulator.selected_entity,
                motion_func=simulator.motion_manager.motion.stay_still
            )

        return None

    def do_current_keys(self, simulator):
        motion_keys = self.pressed_keys & set(self.MOTION_KEY_LIST)

        if not motion_keys:
            simulator.motion_manager.set_motion(
                entity=simulator.selected_entity,
                motion_func=simulator.motion_manager.motion.stay_still
            )
            return

        if self.do_multi_keys(motion_keys, simulator):
            return

        self.do_single_motion_key(motion_keys, simulator)

    @staticmethod
    def do_multi_keys(motion_keys, simulator):
        motion = simulator.motion_manager.motion

        multi_key_motion_map = {
            frozenset([pg.K_UP, pg.K_LEFT]): motion.go_up_left,
            frozenset([pg.K_UP, pg.K_RIGHT]): motion.go_up_right,
            frozenset([pg.K_DOWN, pg.K_LEFT]): motion.go_down_left,
            frozenset([pg.K_DOWN, pg.K_RIGHT]): motion.go_down_right,

            frozenset([pg.K_UP, pg.K_a]): motion.go_up_rotate_anticlockwise,
            frozenset([pg.K_DOWN, pg.K_a]): motion.go_down_rotate_anticlockwise,
            frozenset([pg.K_LEFT, pg.K_a]): motion.go_left_rotate_anticlockwise,
            frozenset([pg.K_RIGHT, pg.K_a]): motion.go_right_rotate_anticlockwise,
            frozenset([pg.K_UP, pg.K_LEFT, pg.K_a]): motion.go_up_left_rotate_anticlockwise,
            frozenset([pg.K_UP, pg.K_RIGHT, pg.K_a]): motion.go_up_right_rotate_anticlockwise,
            frozenset([pg.K_DOWN, pg.K_LEFT, pg.K_a]): motion.go_down_left_rotate_anticlockwise,
            frozenset([pg.K_DOWN, pg.K_RIGHT, pg.K_a]): motion.go_down_right_rotate_anticlockwise,

            frozenset([pg.K_UP, pg.K_d]): motion.go_up_rotate_clockwise,
            frozenset([pg.K_DOWN, pg.K_d]): motion.go_down_rotate_clockwise,
            frozenset([pg.K_LEFT, pg.K_d]): motion.go_left_rotate_clockwise,
            frozenset([pg.K_RIGHT, pg.K_d]): motion.go_right_rotate_clockwise,
            frozenset([pg.K_UP, pg.K_LEFT, pg.K_d]): motion.go_up_left_rotate_clockwise,
            frozenset([pg.K_UP, pg.K_RIGHT, pg.K_d]): motion.go_up_right_rotate_clockwise,
            frozenset([pg.K_DOWN, pg.K_LEFT, pg.K_d]): motion.go_down_left_rotate_clockwise,
            frozenset([pg.K_DOWN, pg.K_RIGHT, pg.K_d]): motion.go_down_right_rotate_clockwise,

            frozenset([pg.K_UP, pg.K_z]): motion.go_up_top_rotate_anticlockwise,
            frozenset([pg.K_DOWN, pg.K_z]): motion.go_down_top_rotate_anticlockwise,
            frozenset([pg.K_LEFT, pg.K_z]): motion.go_left_top_rotate_anticlockwise,
            frozenset([pg.K_RIGHT, pg.K_z]): motion.go_right_top_rotate_anticlockwise,
            frozenset([pg.K_UP, pg.K_LEFT, pg.K_z]): motion.go_up_left_top_rotate_anticlockwise,
            frozenset([pg.K_UP, pg.K_RIGHT, pg.K_z]): motion.go_up_right_top_rotate_anticlockwise,
            frozenset([pg.K_DOWN, pg.K_LEFT, pg.K_z]): motion.go_down_left_top_rotate_anticlockwise,
            frozenset([pg.K_DOWN, pg.K_RIGHT, pg.K_z]): motion.go_down_right_top_rotate_anticlockwise,

            frozenset([pg.K_UP, pg.K_c]): motion.go_up_top_rotate_clockwise,
            frozenset([pg.K_DOWN, pg.K_c]): motion.go_down_top_rotate_clockwise,
            frozenset([pg.K_LEFT, pg.K_c]): motion.go_left_top_rotate_clockwise,
            frozenset([pg.K_RIGHT, pg.K_c]): motion.go_right_top_rotate_clockwise,
            frozenset([pg.K_UP, pg.K_LEFT, pg.K_c]): motion.go_up_left_top_rotate_clockwise,
            frozenset([pg.K_UP, pg.K_RIGHT, pg.K_c]): motion.go_up_right_top_rotate_clockwise,
            frozenset([pg.K_DOWN, pg.K_LEFT, pg.K_c]): motion.go_down_left_top_rotate_clockwise,
            frozenset([pg.K_DOWN, pg.K_RIGHT, pg.K_c]): motion.go_down_right_top_rotate_clockwise,

            frozenset([pg.K_UP, pg.K_w]): motion.go_up_pitch_up,
            frozenset([pg.K_DOWN, pg.K_w]): motion.go_down_pitch_up,
            frozenset([pg.K_LEFT, pg.K_w]): motion.go_left_pitch_up,
            frozenset([pg.K_RIGHT, pg.K_w]): motion.go_right_pitch_up,
            frozenset([pg.K_UP, pg.K_LEFT, pg.K_w]): motion.go_up_left_pitch_up,
            frozenset([pg.K_UP, pg.K_RIGHT, pg.K_w]): motion.go_up_right_pitch_up,
            frozenset([pg.K_DOWN, pg.K_LEFT, pg.K_w]): motion.go_down_left_pitch_up,
            frozenset([pg.K_DOWN, pg.K_RIGHT, pg.K_w]): motion.go_down_right_pitch_up,

            frozenset([pg.K_UP, pg.K_x]): motion.go_up_pitch_down,
            frozenset([pg.K_DOWN, pg.K_x]): motion.go_down_pitch_down,
            frozenset([pg.K_LEFT, pg.K_x]): motion.go_left_pitch_down,
            frozenset([pg.K_RIGHT, pg.K_x]): motion.go_right_pitch_down,
            frozenset([pg.K_UP, pg.K_LEFT, pg.K_x]): motion.go_up_left_pitch_down,
            frozenset([pg.K_UP, pg.K_RIGHT, pg.K_x]): motion.go_up_right_pitch_down,
            frozenset([pg.K_DOWN, pg.K_LEFT, pg.K_x]): motion.go_down_left_pitch_down,
            frozenset([pg.K_DOWN, pg.K_RIGHT, pg.K_x]): motion.go_down_right_pitch_down,
        }

        current_combo = frozenset(motion_keys)
        if current_combo in multi_key_motion_map:
            simulator.motion_manager.set_motion(
                entity=simulator.selected_entity,
                motion_func=multi_key_motion_map[current_combo]
            )
            return True

        return False

    def do_single_motion_key(self, motion_keys, simulator):
        motion = simulator.motion_manager.motion

        single_key_motion_map = {
            pg.K_UP: motion.go_up,
            pg.K_DOWN: motion.go_down,
            pg.K_LEFT: motion.go_left,
            pg.K_RIGHT: motion.go_right,
            pg.K_a: motion.rotate_anticlockwise,
            pg.K_d: motion.rotate_clockwise,
            pg.K_z: motion.top_rotate_anticlockwise,
            pg.K_c: motion.top_rotate_clockwise,
            pg.K_w: motion.pitch_up,
            pg.K_x: motion.pitch_down,
            pg.K_g: motion.ascend,
            pg.K_b: motion.descend,
            pg.K_KP1: motion.up_down_osc,
            pg.K_KP2: motion.left_right_osc,
            pg.K_KP3: motion.diag_osc,
            pg.K_KP4: motion.up_down_osc_rotate,
            pg.K_KP5: motion.left_right_osc_rotate,
            pg.K_KP6: motion.diag_osc_rotate
        }

        for key in self.MOTION_KEY_LIST:
            if key in motion_keys:
                motion_func = single_key_motion_map.get(key)
                if motion_func:
                    simulator.motion_manager.set_motion(
                        entity=simulator.selected_entity,
                        motion_func=motion_func
                    )
                break











