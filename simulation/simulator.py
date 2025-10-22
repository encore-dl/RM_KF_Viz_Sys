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

    def run(self):
        self.update(self.robot_manager.robots)
        self.visualizer.show(self.robot_manager.robots)

    def update(self, robots):
        # 时间在此管理
        t_cur = time.time()
        dt = t_cur - self.t_last
        self.t_last = t_cur

        # 更新相机运动
        self.motion_manager.change_motion(
            entity=self.camera_manager.camera,
            do_motion=self.motion_manager.motion.camera_auto_motion,
            dt=dt
        )
        print(self.camera_manager.camera.world_pos)
        if (self.camera_manager.camera.auto_aiming and  # 相机自瞄
                len(self.robot_manager.robots) != 0):
            self.camera_manager.camera.look_at(self.robot_manager.selected_robot)

        # 更新robot的运动
        if len(self.robot_manager.robots) != 0:
            self.motion_manager.change_motion(self.robot_manager.selected_robot, self.motion_manager.motion.hrz_osc, t_cur)

        # 生产被观测的数据，实际上只有被观测的装甲板，套robot的皮
        observed_armors = self.robot_manager.get_observed_armors()
        if len(observed_armors) != 0:
            targets = self.tracker.track(observed_armors, dt, self.visualizer.camera_screen_center)













