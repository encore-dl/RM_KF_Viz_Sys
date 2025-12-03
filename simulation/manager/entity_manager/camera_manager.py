import numpy as np
import math

from object.entity.camera import Camera


class CameraManager:
    def __init__(self):
        self.cameras = []
        self.selected_camera = None

        self.camera = Camera(
            world_pos=np.array([0., 0., 15.]),
            fov=60,
            max_range=300,
            orient=np.array([0., 0., 0.])
        )