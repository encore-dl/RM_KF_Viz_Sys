import pygame as pg

from simulation.simulator import Simulator
from object.entity.Robot import RobotType


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
                if event.key == pg.K_ESCAPE:
                    running = False
                elif is_second_key:  # 组合按键的判断
                    if first_key == pg.K_BACKSPACE:
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

                        simulator.robot_manage.delete_robot(num)

                    is_second_key = False
                    first_key = None
                elif event.key == pg.K_RETURN:
                    simulator.robot_manage.create_robot(RobotType.Hero)
                elif event.key == pg.K_BACKSPACE:
                    is_second_key = True
                    first_key = event.key

        simulator.run()

        pg.display.flip()
        clock.tick(60)

    pg.quit()


if __name__ == '__main__':
    main()