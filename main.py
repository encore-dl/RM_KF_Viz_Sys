import pygame as pg

from simulation.managers.keyboard_manager import KeyboardManager
from simulation.managers.tracker_manager import TrackerManager
from simulation.simulator import Simulator

from models.demo4.demo_tracker_4 import DemoTracker4
from models.demo5.demo_tracker_5 import DemoTracker5

RESET = True


def main():
    pg.init()
    pg.display.set_caption("RoboMaster KF Visualization System")

    simulator = Simulator()
    keyboard_manager = KeyboardManager(simulator)
    tracker_manager = TrackerManager()

    tracker = DemoTracker5(simulator.robot_manager)
    tracker_manager.set_tracker(tracker)
    tracker_manager.run_tracker_thread()

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

    tracker_manager.thread_shut_down()
    pg.quit()


if __name__ == '__main__':
    while RESET is True:
        RESET = False
        main()
