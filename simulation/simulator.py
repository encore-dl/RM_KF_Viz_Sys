import time

from simulation.visualizer import Visualizer
from simulation.manager.robot_manager import RobotManage
from simulation.manager.camera_manager import CameraManager
from simulation.manager.motion_manager import MotionManager

from object.model.tongji.tracking.tongji_tracker import TongJiTracker


SCREEN_WIDTH = 1500
SCREEN_HEIGHT = 840


class Simulator:
    def __init__(self):
        self.t_last = time.time()

        self.visualizer = Visualizer(SCREEN_WIDTH, SCREEN_HEIGHT)
        self.camera_manager = CameraManager(SCREEN_WIDTH, SCREEN_HEIGHT)
        self.robot_manager = RobotManage(self.camera_manager.camera)
        self.motion_manager = MotionManager()
        self.tracker = TongJiTracker()  # 暂时使用第一个robot
        
        self.selected_entity = None

    def run(self):
        self.update()
        self.visualizer.show(
            self.robot_manager.robots,
            self.robot_manager.obsrv_armors,
            self.tracker,
            self.camera_manager.camera
        )

    def update(self):
        # 时间在此管理
        t_cur = time.time()
        dt = t_cur - self.t_last
        self.t_last = t_cur

        # 相机自瞄
        if (self.camera_manager.camera.auto_aiming and
                len(self.robot_manager.robots) != 0):
            self.camera_manager.camera.look_at(self.robot_manager.selected_robot.world_pos)

        self.motion_manager.update(dt, t_cur)

        # 生产被观测的数据，实际上只有被观测的装甲板，套robot的皮
        self.robot_manager.get_obsrv_armors()
        if len(self.robot_manager.obsrv_armors) != 0:
            self.tracker.track(self.robot_manager.obsrv_armors, dt, self.visualizer.camera_screen_center)
    
    def select_entity(self, selected_type, entity_number=None):
        if selected_type == 'robot':
            if entity_number is not None and 0 <= entity_number < len(self.robot_manager.robots):
                self.selected_entity = self.robot_manager.robots[entity_number]
            else:
                pass
        elif selected_type == 'camera':
            self.selected_entity = self.camera_manager.camera













