import pygame as pg

from simulation.simulator import Simulator
from object.entity.robot import RobotType


def main():
    pg.init()

    pg.display.set_caption("RoboMaster KF Visualization System")
    clock = pg.time.Clock()

    simulator = Simulator()

    running = True
    first_key = None
    is_second_key = False
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:  # esc 退出
                    running = False
                elif is_second_key:  # 组合按键的判断
                    if first_key == pg.K_BACKSPACE:  # 删除 robot 编号
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
                    elif first_key == pg.K_a:  # auto + direction
                        if event.key == pg.K_UP:
                            simulator.motion_manager.set_auto_step(auto_pos_step=[0., -1., 0.])
                        elif event.key == pg.K_DOWN:
                            simulator.motion_manager.set_auto_step(auto_pos_step=[0., 1., 0.])
                        elif event.key == pg.K_LEFT:
                            simulator.motion_manager.set_auto_step(auto_pos_step=[-1., 0., 0.])
                        elif event.key == pg.K_RIGHT:
                            simulator.motion_manager.set_auto_step(auto_pos_step=[1., 0., 0.])

                    is_second_key = False
                    first_key = None
                elif event.key == pg.K_1:  # 1 为 robots 的第一个 robot
                    simulator.select_entity('robot', 0)
                elif event.key == pg.K_2:  # 2 为 camera
                    simulator.select_entity('camera')
                elif event.key == pg.K_RETURN:  # 回车 生成robot
                    simulator.robot_manager.create_robot(RobotType.Hero)
                elif event.key == pg.K_BACKSPACE:  # 删除
                    is_second_key = True
                    first_key = event.key
                elif event.key == pg.K_a:  # a -> auto move
                    is_second_key = True
                    first_key = event.key
                if event.key == pg.K_UP:
                    simulator.motion_manager.set_auto_step(auto_pos_step=[0., -1., 0.])
                elif event.key == pg.K_DOWN:
                    simulator.motion_manager.set_auto_step(auto_pos_step=[0., 1., 0.])
                elif event.key == pg.K_LEFT:
                    simulator.motion_manager.set_auto_step(auto_pos_step=[-1., 0., 0.])
                elif event.key == pg.K_RIGHT:
                    simulator.motion_manager.set_auto_step(auto_pos_step=[1., 0., 0.])
                elif event.key == pg.K_SPACE:  # 空格 暂停
                    simulator.motion_manager.set_auto_step(auto_pos_step=[0., 0., 0.])
            elif event.type == pg.KEYUP:
                print('key up')

        simulator.run()

        pg.display.flip()
        clock.tick(60)

    pg.quit()


if __name__ == '__main__':
    main()