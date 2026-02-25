import numpy as np
import time

from core.algorithms.math import euler_to_rotation_matrix
from core.algorithms.perspective_n_point.perspective_n_point import solve_pnp_core
from simulation.event_bus import event_bus
from simulation.dataflow import ArmorObservation, Observation, PnPResult


class SensorManager:
    def __init__(self, camera_manager):
        self.camera_manager = camera_manager
        self.pos_noise_sigma = 0.00
        self.rpy_noise_sigma = 0.0
        self.pixel_noise_sigma = 0.

    def get_obs(self, robots):
        camera = self.camera_manager.selected_camera
        
        obs_armors = []
        pnp_results = []

        for robot in robots:
            for armor in robot.armors:
                # 检查法向量是否朝向相机
                if not camera.is_armor_visible(armor.world_pos, robot.world_pos):
                    continue
                # 检查装甲板中心是否在相机视野内
                if camera.world_to_pixel(armor.world_pos) is None:
                    continue

                # 计算角点和像素点
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

                # PnP求解
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
                    armor_type=armor.armor_size,
                    image_points_2d=ordered_pixels,
                    head_yaw=head_yaw,
                    trans_head2world=T_head2world,
                    rot_head2world=R_head2world,
                    trans_pnp2head=T_pnp2head,
                    rot_pnp2head=R_pnp2head,
                    use_plus_pnp=True
                )

                pos_noise = np.random.normal(0, self.pos_noise_sigma, 3)
                rpy_noise = np.random.normal(0, self.rpy_noise_sigma, 3)

                obs_armor = ArmorObservation(
                    armor_id=armor.armor_id,
                    robot_type=armor.robot_type,
                    world_pos=calc_pos + pos_noise,
                    world_rpy=calc_rpy + rpy_noise,
                    armor_size=armor.armor_size,
                    pixel_points=ordered_pixels
                )
                obs_armors.append(obs_armor)

                if calc_pos is not None and not np.isnan(calc_pos).any():
                    pnp_result = PnPResult(
                        pnp_pos=calc_pos,
                        pnp_rpy=calc_rpy,
                        true_pos=armor.world_pos,
                        pixel_points=ordered_pixels,
                        world_corners=world_corners,
                        armor_size=armor.armor_size
                    )
                    pnp_results.append(pnp_result)

        obs = Observation(obs_armors=obs_armors, timestamp=time.time())
        event_bus.publish('obs', obs)
        if pnp_results:
            event_bus.publish('pnp', pnp_results)




