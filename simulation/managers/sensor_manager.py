import numpy as np
import time

from core.algorithms.math import euler_to_rotation_matrix, world_to_robot
from core.algorithms.perspective_n_point.perspective_n_point import solve_pnp_core
from core.entities.property.robot_type import RobotType
from simulation.event_bus import event_bus
from simulation.dataflow import ArmorObservation, Observation


class SensorManager:
    def __init__(self, robot_manager, device_manager):
        self.robot_manager = robot_manager
        self.device_manager = device_manager
        self.pos_noise_sigma = 0.0
        self.rpy_noise_sigma = 0.0
        self.pixel_noise_sigma = 0.05

    def get_obs(self):
        if not self.robot_manager.robots or self.robot_manager.viewing_robot is None:
            return
        viewing = self.robot_manager.viewing_robot
        camera = viewing.get_camera()

        obs_armors = []
        obs_armors.extend(self._get_robot_obs(viewing, camera))
        obs_armors.extend(self._get_outpost_obs(viewing, camera))

        if not obs_armors:
            # print("no obs")
            return

        obs = Observation(
            obs_armors=obs_armors,
            timestamp=time.time()
        )
        event_bus.publish('obs', obs)

    def _get_robot_obs(self, viewing, camera):
        obs_list = []
        for robot in self.robot_manager.robots:
            if robot == viewing:  # 不自瞄自己
                continue
            for armor in robot.get_armors():
                obs_armor = self._process_armor(armor, camera, robot.chassis.world_pos, viewing.chassis)
                if obs_armor is not None:
                    obs_list.append(obs_armor)
        return obs_list

    def _get_outpost_obs(self, viewing, camera):
        obs_list = []
        outpost = self.device_manager.outpost
        if outpost is None:
            return obs_list
        for armor in outpost.armors:
            obs_armor = self._process_armor(armor, camera, outpost.world_pos, viewing.chassis)
            if obs_armor is not None:
                obs_list.append(obs_armor)
        return obs_list

    def _process_armor(self, armor, camera, target_pos, viewing_chassis):
        if not camera.is_armor_visible(armor.world_pos, target_pos):
            return None
        if camera.world_to_pixel(armor.world_pos) is None:
            return None

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
                return None
            noise_uv = np.random.normal(0, self.pixel_noise_sigma)
            noisy_uv = np.array([uv[0] + noise_uv, uv[1] + noise_uv], dtype=np.float32)
            pixel_points.append(noisy_uv)

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
            use_plus_pnp=True
        )

        if calc_pos is None:
            return None

        pos_noise = np.random.normal(0, self.pos_noise_sigma, 3)
        rpy_noise = np.random.normal(0, self.rpy_noise_sigma, 3)

        rel_pos = world_to_robot(calc_pos, viewing_chassis)
        obs_armor = ArmorObservation(
            armor_id=armor.armor_id,
            robot_type=armor.robot_type,
            rel_pos=rel_pos + pos_noise,
            rel_rpy=calc_rpy + rpy_noise,
            armor_size=armor.armor_size,
            pixel_points=ordered_pixels
        )

        return obs_armor



