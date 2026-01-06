import numpy as np

from object.entity.rigid.camera import Camera


class CameraManager:
    def __init__(self):
        self.cameras = []
        self.selected_camera = None

        self.camera = Camera(
            world_pos=np.array([0., 0., 0.15]),
            world_rpy=np.array([0., 0., 0.]),
            fov=360,
            max_range=10,
        )

    def switch_auto_aiming(self):
        self.camera.auto_aiming = not self.camera.auto_aiming


