import pygame.time as pgtime

from simulation.manager.entity_manager.robot_manager import RobotManager
from simulation.manager.entity_manager.camera_manager import CameraManager
from simulation.manager.entity_manager.motion_manager import MotionManager
from simulation.manager.system_manager.visualization_manager import VisualizationManager
from simulation.manager.system_manager.tracker_thread_manager import TrackerThreadManager

from object.model.tongji.tracking.tongji_tracker import TongJiTracker


SCREEN_WIDTH = 1500
SCREEN_HEIGHT = 840


class Simulator:
    def __init__(self):
        self.clock = pgtime.Clock()
        self.last_t = pgtime.get_ticks()
        self.simulator_fps = 500.

        self.camera_manager = CameraManager()
        self.robot_manager = RobotManager(self.camera_manager.camera)
        self.motion_manager = MotionManager()
        self.visualization_manager = VisualizationManager(SCREEN_WIDTH, SCREEN_HEIGHT)
        self.tracker_thread_manager = TrackerThreadManager()

        self.selected_entity = None

    def run_simulator(self):
        self.update()

        output_data = self.tracker_thread_manager.get_tracker_output()
        if output_data is None:
            tracker = None
        else:
            tracker = output_data[0]

        self.visualization_manager.show(
            self.robot_manager.robots,
            self.robot_manager.obsrv_armors_with_t[0],
            tracker,
            self.camera_manager.camera
        )

    def update(self):
        curr_t = pgtime.get_ticks()
        dt = (curr_t - self.last_t) / 1000.
        self.last_t = curr_t

        # 相机自瞄
        if (self.camera_manager.camera.auto_aiming and
                len(self.robot_manager.robots) != 0):
            self.camera_manager.camera.look_at(self.robot_manager.selected_robot.world_pos)

        self.motion_manager.update(dt, curr_t)

        # 生产被观测的数据，实际上只有被观测的装甲板
        self.robot_manager.get_obsrv_armors(self.camera_manager.camera)
        if len(self.robot_manager.obsrv_armors_with_t[0]) != 0:
            self.tracker_thread_manager.put_tracker_input(
                self.robot_manager.obsrv_armors_with_t
            )

        self.clock.tick(self.simulator_fps)

    def select_entity(self, selected_type, entity_number=None):
        if selected_type == 'robot':
            if entity_number is not None and 0 <= entity_number < len(self.robot_manager.robots):
                self.selected_entity = self.robot_manager.robots[entity_number]
            else:
                pass
        elif selected_type == 'camera':
            self.selected_entity = self.camera_manager.camera













