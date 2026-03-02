from core.entities.rigid.robot import Robot
from core.entities.property.robot_type import RobotType
from models.demo3.demo_tjurm_model import DemoTJURMModel


class TrackedRobot:
    def __init__(self, robot_type):
        self.robot = Robot(robot_type)
        self.model = DemoTJURMModel()


class DemoTJURMTracker:
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

    def track(self, obs_armors, dt, timestamp):
        if not obs_armors:
            self.is_tracking = False
            self.pred_pos = []
            return

        for obs_armor in obs_armors:
            self.tracked_robots[self.robot_type_map[obs_armor.robot_type]].model.push(obs_armor, timestamp)

        for tracked_robot in self.tracked_robots:
            tracked_robot.model.update()

        self.is_tracking = True

        tracked_robot_now = self.tracked_robots[self.robot_type_map[obs_armors[0].robot_type]]
        self.pred_pos = tracked_robot_now.model.get_pred_pos()

        return True







