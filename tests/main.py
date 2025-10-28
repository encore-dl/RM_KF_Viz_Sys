import pygame as pg

from simulation.simulator import Simulator
from object.entity.robot import RobotType


def main():
    pg.init()

    pg.display.set_caption("RoboMaster KF Visualization System")
    clock = pg.time.Clock()

    simulator = Simulator()

    running = True
    last_key = None
    is_second_key = False
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:  # esc 退出
                    running = False
                elif is_second_key:  # 组合按键的判断
                    if last_key == pg.K_BACKSPACE:  # 删除 robot 编号
                        num = None
                        if event.key == pg.K_0:
                            num = 0
                        elif event.key == pg.K_1:
                            num = 1
                        elif event.key == pg.K_2:
                            num = 2
                        elif event.key == pg.K_3:
                            num = 3
                        elif event.key == pg.K_4:
                            num = 4
                        elif event.key == pg.K_5:
                            num = 5
                        else:
                            continue
                        simulator.robot_manager.delete_robot(num)
                    is_second_key = False
                elif event.key == pg.K_1:  # 1 为 robots 的第一个 robot
                    simulator.select_entity('robot', 0)
                elif event.key == pg.K_2:  # 2 为 camera
                    simulator.select_entity('camera')
                elif event.key == pg.K_RETURN:  # 回车 生成robot
                    simulator.robot_manager.create_robot(RobotType.Hero)
                elif event.key == pg.K_BACKSPACE:  # 删除
                    is_second_key = True
                elif event.key == pg.K_a:
                    simulator.camera_manager.camera.auto_aiming = not simulator.camera_manager.camera.auto_aiming
                selected_entity_move(event.key, simulator)
            elif event.type == pg.KEYUP and is_key_up_still(last_key):
                simulator.motion_manager.set_motion(
                    entity=simulator.selected_entity,
                    motion_func=simulator.motion_manager.motion.stay_still
                )

            if hasattr(event, 'key'):
                last_key = event.key

        simulator.run()

        pg.display.flip()
        clock.tick(60)

    pg.quit()


def is_key_up_still(key):
    if (
        key == pg.K_UP or
        key == pg.K_DOWN or
        key == pg.K_LEFT or
        key == pg.K_RIGHT or
        key == pg.K_KP4 or
        key == pg.K_KP6 or
        key == pg.K_KP1 or
        key == pg.K_KP3
    ):
        return True


# 被选中的实体进行运动更新
def selected_entity_move(key, simulator):
    if key == pg.K_UP:
        print('go up')
        simulator.motion_manager.set_motion(
            entity=simulator.selected_entity,
            motion_func=simulator.motion_manager.motion.go_up
        )
    elif key == pg.K_DOWN:
        simulator.motion_manager.set_motion(
            entity=simulator.selected_entity,
            motion_func=simulator.motion_manager.motion.go_down
        )
    elif key == pg.K_LEFT:
        simulator.motion_manager.set_motion(
            entity=simulator.selected_entity,
            motion_func=simulator.motion_manager.motion.go_left
        )
    elif key == pg.K_RIGHT:
        simulator.motion_manager.set_motion(
            entity=simulator.selected_entity,
            motion_func=simulator.motion_manager.motion.go_right
        )
    elif key == pg.K_KP4:
        simulator.motion_manager.set_motion(
            entity=simulator.selected_entity,
            motion_func=simulator.motion_manager.motion.rotate_anticlockwise
        )
    elif key == pg.K_KP6:
        simulator.motion_manager.set_motion(
            entity=simulator.selected_entity,
            motion_func=simulator.motion_manager.motion.rotate_clockwise
        )
    elif key == pg.K_KP1:
        simulator.motion_manager.set_motion(
            entity=simulator.selected_entity,
            motion_func=simulator.motion_manager.motion.top_rotate_anticlockwise
        )
    elif key == pg.K_KP3:
        simulator.motion_manager.set_motion(
            entity=simulator.selected_entity,
            motion_func=simulator.motion_manager.motion.top_rotate_clockwise
        )
    elif key == pg.K_SPACE:  # 空格 暂停
        simulator.motion_manager.set_motion(
            entity=simulator.selected_entity,
            motion_func=simulator.motion_manager.motion.stay_still
        )


if __name__ == '__main__':
    main()