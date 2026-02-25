from core.entities.rigid.camera import Camera
import numpy as np


class CameraManager:
    def __init__(self):
        self.cameras = []
        self.selected_camera = None

    def create_camera(self, world_pos=None, world_rpy=None, fov=60, max_range=10):
        if world_pos is None:
            world_pos = np.array([0., 0., 0.15])
        if world_rpy is None:
            world_rpy = np.array([0., 0., 0.])
        camera = Camera(
            world_pos=world_pos,
            world_rpy=world_rpy,
            fov=fov,
            max_range=max_range
        )
        self.cameras.append(camera)
        if self.selected_camera is None:
            self.selected_camera = camera
        return camera

    def delete_camera(self, index):
        if 0 <= index < len(self.cameras):
            if self.selected_camera == self.cameras[index]:
                self.selected_camera = None
            self.cameras.pop(index)
            if self.cameras and self.selected_camera is None:
                self.selected_camera = self.cameras[0]
        else:
            print(f"Camera index {index} out of range")

    def next_camera(self):
        if not self.cameras:
            return None
        if self.selected_camera is None:
            self.selected_camera = self.cameras[0]
        else:
            idx = self.cameras.index(self.selected_camera)
            idx = (idx + 1) % len(self.cameras)
            self.selected_camera = self.cameras[idx]
        return self.selected_camera


