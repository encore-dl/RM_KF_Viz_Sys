import pygame as pg

from simulation.manager.system_manager.keyboard_manager import KeyboardManager
from simulation.simulator import Simulator

from object.model.tongji.tracking.tongji_tracker import TongJiTracker
from object.model.tjurm.tracking.tjurm_tracker import TJURMTracker


def main():
    pg.init()
    pg.display.set_caption("RoboMaster KF Visualization System")

    simulator = Simulator()
    keyboard_manager = KeyboardManager()

    simulator.tracker_manager.set_tracker(TJURMTracker())
    simulator.tracker_manager.run_tracker_thread()

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
        simulator.run_simulator()

        pg.display.flip()

    pg.quit()


if __name__ == '__main__':
    main()