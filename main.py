import pygame as pg

from simulation.managers.system_manager.keyboard_manager import KeyboardManager
from simulation.simulator import Simulator

from models.demo.demo_tracker import DemoTracker
from models.demo2.demo_tracker_2 import DemoTracker2
from models.demo3.demo_tracker_3 import DemoTJURMTracker
from models.imm1.imm_tracker_1 import IMMTracker1
from models.demo4.demo_tracker_4 import DemoTracker4

RESET = True


def main():
    pg.init()
    pg.display.set_caption("RoboMaster KF Visualization System")

    simulator = Simulator()
    keyboard_manager = KeyboardManager(simulator)

    simulator.tracker_manager.set_tracker(DemoTracker4(simulator.camera_manager))
    # simulator.tracker_manager.set_tracker(DemoTracker2())
    simulator.tracker_manager.run_tracker_thread()

    running = True
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type in (pg.KEYDOWN, pg.KEYUP):
                result = keyboard_manager.handle_event(event)
                if result == 'escape':
                    running = False
                elif result == 'reset':
                    global RESET
                    RESET = True
                    return

        simulator.run_simulator()

        pg.display.flip()

    pg.quit()


if __name__ == '__main__':
    while RESET is True:
        RESET = False
        main()

