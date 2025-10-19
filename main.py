import pygame as pg
from object.abstract_object.Simulator import Simulator

SCREEN_WIDTH = 1800
SCREEN_HEIGHT = 800


def main():
    pg.init()

    screen = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pg.display.set_caption("RoboMaster KF Visualization System")
    clock = pg.time.Clock()

    simulator = Simulator(screen)

    running = True
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    running = False
                elif event.key == pg.K_SPACE:
                    # simulator
                    pass

        simulator.run()

        pg.display.flip()
        clock.tick(60)

    pg.quit()


if __name__ == '__main__':
    main()