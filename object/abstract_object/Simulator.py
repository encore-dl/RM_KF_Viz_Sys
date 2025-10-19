import time

class Simulator:
    def __init__(self, screen):
        self.screen = screen

        self.t_last = time.time()

    def run(self):
        self.update()
        self.show()

    def update(self):
        t_cur = time.time()
        dt = t_cur - self.t_last
        self.t_last = t_cur

    def show_main(self):
        pass

    def show_camera(self):
        pass

    def show_info(self):
        pass

    def show(self):
        self.show_main()
        self.show_camera()
        self.show_info()




