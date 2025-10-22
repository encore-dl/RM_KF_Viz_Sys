import numpy as np

from object.entity.robot import RobotType
from algorithms.filter.extended_kalman_filter import ExtendedKalmanFilter
from utils.math_tool import limit_rad


# 状态向量元素: [x, vx, y, vy, z, vz, agl, omg, r, dr, dz] 11维向量
# 观测向量元素: [roll, pitch, yaw, agl]
class TongJiModel:
    def __init__(self):
        self.armor = None
        self._ekf = None

        self.update_count = 0
        self.is_converged = False

    def init_model(self, armor):
        self.armor = armor

        x0 = np.array([
            self.armor.world_pos[0], 0,
            self.armor.world_pos[1], 0,
            self.armor.world_pos[2], 0,
            self.armor.world_rpy[0], 0,
            self.armor.radius, 0, 0,
        ])
        P0_diag = np.array([1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1])
        P0 = np.diag(P0_diag)

        def x_add(a, b):
            c = a + b
            c[6] = limit_rad(c[6])
            return c

        self._ekf = ExtendedKalmanFilter(x0, P0, x_add)

    def predict(self, dt):
        F = np.eye(11)
        F[0, 1] = dt  # x = x + vx*dt
        F[2, 3] = dt  # y = y + vy*dt
        F[4, 5] = dt  # z = z + vz*dt
        F[6, 7] = dt  # a = a + w*dt

        v1, v2 = (10, 0.1) if self.armor.robot_type == RobotType.Outpost else (100, 400)
        p_noise = dt ** 4 / 4
        pv_noise = dt ** 3 / 2
        v_noise = dt ** 2
        Q = np.zeros((11, 11))
        Q[0:2, 0:2] = [  # x, vx
            [p_noise * v1, pv_noise * v1],
            [pv_noise * v1, v_noise * v1]
        ]
        Q[2:4, 2:4] = [  # y, vy
            [p_noise * v1, pv_noise * v1],
            [pv_noise * v1, v_noise * v1]
        ]
        Q[4:6, 4:6] = [  # z, vz
            [p_noise * v1, pv_noise * v1],
            [pv_noise * v1, v_noise * v1]
        ]
        Q[6:8, 6:8] = [  # agl, omg
            [p_noise * v2, pv_noise * v2],
            [pv_noise * v2, v_noise * v2]
        ]

        # 人工设置前哨战的最高转速
        if self.is_converged and self.robot_type == RobotType.Outpost and abs(self._ekf.x[7]) > 2:
            self._ekf.x[7] = 2.51 if self._ekf.x[7] > 0 else -2.51

        self._ekf.predict(F, Q)

        self._ekf.x[6] = limit_rad(self._ekf.x[6])

    def update(self, armor):
        self.update_count += 1

        # H = self.h_jacobian()
    
    # def

    def h_jacobian(self):
        # x = self._ekf.x
        # agl = limit_rad(x[6] + self.robot.armors[] * 2 * 1)
        # angle = self.robot.world_rpy[0]
        #
        # for armor in self.robot.armors:
            pass


    def diverged(self):
        r_ok = 0.05 < self._ekf.x[8] < 0.5
        l_ok = 0.05 < (self._ekf.x[8] + self._ekf.x[9]) < 0.5
        return not (r_ok and l_ok)

    def converged(self):
        if self.robot.robot_type != RobotType.Outpost and self.update_count > 3 and not self.diverged():
            self.is_converged = True
        if self.robot.robot_type == RobotType.Outpost and self.update_count > 10 and not self.diverged():
            self.is_converged = True

        return self.is_converged

    def get_ekf(self):
        return self._ekf









