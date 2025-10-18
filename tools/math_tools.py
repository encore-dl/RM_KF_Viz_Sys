def world_to_screen(self, world_xy):
    """将世界XY坐标转换为屏幕坐标"""
    screen_x = self.screen_center[0] + world_xy[0] * self.world_scale
    screen_y = self.screen_center[1] - world_xy[1] * self.world_scale  # Y轴翻转
    return int(screen_x), int(screen_y)