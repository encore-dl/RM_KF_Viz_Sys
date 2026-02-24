from models.tjurm.tjurm_model import TJURMModel

from core.entities.rigid.robot import Robot, RobotType


class TrackedRobot:
    def __init__(self, robot_type):
        self.robot = Robot(robot_type)
        self.model = TJURMModel()


class TJURMTracker:
    def __init__(self):
        self.tracked_robots = [
            TrackedRobot(RobotType.Hero),
            TrackedRobot(RobotType.Infantry),
            TrackedRobot(RobotType.Sentry),
            TrackedRobot(RobotType.Engineer),
        ]
        self.robot_type_map = {
            RobotType.Hero: 0,
            RobotType.Infantry: 1,
            RobotType.Sentry: 2,
            RobotType.Engineer: 3
        }

        self.is_tracking = False
        self.pred_pos = []
        self.state_vecs = None

    def track(self, obsrv_armors, dt, t_stamp):
        for obsrv_armor in obsrv_armors:
            self.tracked_robots[self.robot_type_map[obsrv_armor.robot_type]].model.push(obsrv_armor, t_stamp)

        for tracked_robot in self.tracked_robots:
            tracked_robot.model.update()

        self.is_tracking = True

        tracked_robot_now = self.tracked_robots[self.robot_type_map[obsrv_armors[0].robot_type]]
        self.pred_pos = tracked_robot_now.model.get_pred_pos()
        self.state_vecs = [
            tracked_robot_now.model.antitop.main_model.x,
            tracked_robot_now.model.antitop.center_model.x,
            tracked_robot_now.model.antitop.omega_model.x,
        ]

        return True






