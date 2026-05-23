import numpy as np
from core.entities.rigid.outpost import Outpost
from core.entities.rigid.rune import Rune


class DeviceManager:
    def __init__(self, motion_manager):
        self.outpost = None
        self.rune = None
        self.motion_manager = motion_manager

    def create_device(self, name_str):
        if name_str == 'Outpost':
            self.outpost = Outpost()
            self.motion_manager.add_entity(self.outpost)
        elif name_str == 'Rune':
            self.rune = Rune()
            self.motion_manager.add_entity(self.rune)

    def delete_device(self, name_str):
        if name_str == 'Outpost' and self.outpost is not None:
            self.motion_manager.remove_entity(self.outpost)
            del self.outpost
            self.outpost = None
        elif name_str == 'Rune' and self.rune is not None:
            self.motion_manager.remove_entity(self.rune)
            del self.rune
            self.rune = None



