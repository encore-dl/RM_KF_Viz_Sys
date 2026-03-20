import numpy as np
import time

from core.algorithms.math import euler_to_rotation_matrix, world_to_robot
from core.algorithms.perspective_n_point.perspective_n_point import solve_pnp_core
from simulation.event_bus import event_bus
from simulation.dataflow import ArmorObservation, Observation


class SensorManager:
    def __init__(self, robot_manager):
        self.robot_manager = robot_manager
        self.pos_noise_sigma = 0.0
        self.rpy_noise_sigma = 0.0
        self.pixel_noise_sigma = 0.

    def get_obs(self):
        if not self.robot_manager.robots or self.robot_manager.viewing_robot is None:
            return
        viewing = self.robot_manager.viewing_robot
        camera = viewing.get_camera()

        obs_armors = []

        for robot in self.robot_manager.robots:
            if robot == viewing:
                continue  # 不自瞄自己
            for armor in robot.get_armors():
                if not camera.is_armor_visible(armor.world_pos, robot.chassis.world_pos):
                    continue
                if camera.world_to_pixel(armor.world_pos) is None:
                    continue

                # 计算角点
                if armor.armor_size == 'large':
                    w, h = armor.light_bar_interval, armor.light_bar_length
                else:
                    w, h = armor.light_bar_interval, armor.light_bar_length

                local_corners = np.array([
                    [0, w / 2, h / 2],
                    [0, -w / 2, h / 2],
                    [0, -w / 2, -h / 2],
                    [0, w / 2, -h / 2]
                ])

                R_armor = euler_to_rotation_matrix(armor.world_rpy)
                world_corners = (R_armor @ local_corners.T).T + armor.world_pos

                pixel_points = []
                for world_corner in world_corners:
                    uv = camera.world_to_pixel(world_corner)
                    if uv is None:
                        pixel_points = None
                        break
                    noise_uv = np.random.normal(0, self.pixel_noise_sigma)
                    noisy_uv = np.array([uv[0] + noise_uv, uv[1] + noise_uv], dtype=np.float32)
                    pixel_points.append(noisy_uv)

                if pixel_points is None:
                    continue

                ordered_pixels = np.array(pixel_points, dtype=np.float32)

                K = camera.get_intrinsic_matrix()
                D = np.zeros(5)
                R_head2world = euler_to_rotation_matrix(camera.world_rpy)
                T_head2world = camera.world_pos
                R_pnp2head = camera.R_body_to_optical.T
                T_pnp2head = np.zeros(3)
                head_yaw = camera.world_rpy[2]

                calc_pos, calc_rpy = solve_pnp_core(
                    camera_k=K,
                    camera_dist=D,
                    armor_size=armor.armor_size,
                    image_points_2d=ordered_pixels,
                    head_yaw=head_yaw,
                    trans_head2world=T_head2world,
                    rot_head2world=R_head2world,
                    trans_pnp2head=T_pnp2head,
                    rot_pnp2head=R_pnp2head,
                    use_plus_pnp=False
                )

                pos_noise = np.random.normal(0, self.pos_noise_sigma, 3)
                rpy_noise = np.random.normal(0, self.rpy_noise_sigma, 3)

                if calc_pos is not None:
                    rel_pos = world_to_robot(calc_pos, viewing.chassis)
                    obs_armor = ArmorObservation(
                        armor_id=armor.armor_id,
                        robot_type=armor.robot_type,
                        rel_pos=rel_pos + pos_noise,
                        rel_rpy=calc_rpy + rpy_noise,
                        armor_size=armor.armor_size,
                        pixel_points=ordered_pixels
                    )
                    obs_armors.append(obs_armor)

        obs = Observation(obs_armors=obs_armors, timestamp=time.time())
        event_bus.publish('obs', obs)



