import pygame as pg

from simulation.manager.keyboard_manager import KeyboardManager
from simulation.simulator import Simulator


def main():
    pg.init()
    pg.display.set_caption("RoboMaster KF Visualization System")
    clock = pg.time.Clock()

    simulator = Simulator()
    keyboard_manager = KeyboardManager()

    running = True
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type in (pg.KEYDOWN, pg.KEYUP):
                result = keyboard_manager.handle_event(event, simulator)
                if result == 'escape':
                    running = False

        keyboard_manager.update(simulator)
        simulator.run()

        pg.display.flip()
        clock.tick(60)

    pg.quit()


if __name__ == '__main__':
    main()