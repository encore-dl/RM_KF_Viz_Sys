import numpy as np

from object.entity.camera import Camera


class CameraManager:
    def __init__(self, camera_screen_width, camera_screen_height):
        self.cameras = []
        self.selected_camera = None

        self.camera_screen_width = camera_screen_width
        self.camera_screen_height = camera_screen_height

        self.camera = Camera(
            (self.camera_screen_width, self.camera_screen_height),
            np.array([0., 0., 0.]),
            60,
            10,
            np.array([0., 0., 0.])
        )