import pygame.time as pgtime

from simulation.managers.entity_manager.robot_manager import RobotManager
from simulation.managers.entity_manager.sensor_manager import SensorManager
from simulation.managers.entity_manager.motion_manager import MotionManager
from simulation.managers.system_manager.visualization_manager import VisualizationManager
from simulation.managers.system_manager.tracker_manager import TrackerManager

SCREEN_WIDTH = 1500
SCREEN_HEIGHT = 840


class Simulator:
    def __init__(self):
        self.clock = pgtime.Clock()
        self.last_t = pgtime.get_ticks()
        self.simulator_fps = 500.

        self.selected_entity = None

        self.camera_manager = SensorManager()
        self.robot_manager = RobotManager(self.camera_manager.camera)
        self.motion_manager = MotionManager()
        self.visualization_manager = VisualizationManager(SCREEN_WIDTH, SCREEN_HEIGHT)
        self.tracker_manager = TrackerManager()

    def run_simulator(self):
        self.update()

        # 生产被观测的数据，实际上只有被观测的装甲板
        self.camera_manager.get_obs(self.robot_manager.robots)

        self.visualization_manager.show(
            self.robot_manager.robots,
            self.camera_manager.camera,
        )

        # self.draw_psi_d_graph()

    def update(self):
        curr_t = pgtime.get_ticks()
        dt = (curr_t - self.last_t) / 1000.
        self.last_t = curr_t

        # 相机自瞄
        if (self.camera_manager.camera.auto_aiming and
                len(self.robot_manager.robots) != 0):
            self.camera_manager.camera.look_at(self.robot_manager.selected_robot.world_pos)

        self.motion_manager.update(dt, curr_t)

        self.clock.tick(self.simulator_fps)

    def select_entity(self, selected_type, entity_number=None):
        if selected_type == 'robot':
            if entity_number is not None and 0 <= entity_number < len(self.robot_manager.robots):
                self.selected_entity = self.robot_manager.robots[entity_number]
            else:
                pass
        elif selected_type == 'camera':
            self.selected_entity = self.camera_manager.camera

    # def draw_psi_d_graph(self):
    #     if len(self.camera_manager.obs_data_with_t[0]) != 0:
    #         psi = self.camera_manager.obs_data_with_t[0][0].world_rpy[2]
    #     if len(self.robot_manager.robots) != 0:
    #         d = np.linalg.norm(self.robot_manager.robots[0].world_pos - self.camera_manager.camera.world_pos)













