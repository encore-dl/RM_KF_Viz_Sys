import threading
import queue
import time


class TrackerManager:
    def __init__(self):
        self.running = True

        self._input_queue = queue.Queue(maxsize=10)
        self._output_queue = queue.Queue(maxsize=5)

        self._tracker = None
        self.tracker_thread = None

        self.target_thread_fps = 300.

        self.fps = 0.
        self.fps_window_len = 10

    def set_tracker(self, tracker):
        self._tracker = tracker

    def run_tracker_thread(self):
        self.tracker_thread = threading.Thread(
            target=self._tracker_thread_func,
            daemon=True
        )
        self.tracker_thread.start()

    def _tracker_thread_func(self):
        dts = []
        last_t = time.time()

        last_input_t_stamp = None

        while self.running:
            curr_t = time.time()
            dts.append(curr_t - last_t)
            last_t = curr_t
            if len(dts) == self.fps_window_len:
                self.fps = len(dts) / sum(dts)
                dts = []

            thread_start_t = time.time()

            try:
                # 获取输入数据
                input_data = self._input_queue.get(timeout=0.5)
                obsrv_armors, input_t_stamp = input_data
                if last_input_t_stamp is None:
                    input_dt = 0.01
                else:
                    input_dt = input_t_stamp - last_input_t_stamp
                    # input_dt = max(min(input_dt, 0.1), 0.001)
                last_input_t_stamp = input_t_stamp

                self._tracker.track(obsrv_armors, input_dt, input_t_stamp)  # 调用track

                output_data = (
                    self._tracker.is_tracked,
                    self._tracker.pred_pos,
                    self._tracker.status,
                    self.fps,
                    time.time()  # output 的时间戳
                )

                try:
                    self._output_queue.put_nowait(output_data)
                except queue.Full:
                    self._output_queue.get_nowait()
                    self._output_queue.put_nowait(output_data)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Tracker thread error: {e}")
                break

            thread_dt = time.time() - thread_start_t
            sleep_t = 1/self.target_thread_fps - thread_dt
            if sleep_t > 0:
                time.sleep(sleep_t)

    def put_tracker_input(self, obsrv_armor_with_t):
        if not self.running:
            return False

        input_data = obsrv_armor_with_t
        try:
            self._input_queue.put_nowait(input_data)
            return True
        except queue.Full:  # 队列满了，说明tracker的处理能力弱，但simu模拟真实世界不能被阻塞，因而相当于直接
            self._input_queue.get_nowait()
            self._input_queue.put_nowait(input_data)
            return True

    def get_tracker_output(self, max_delay=0.1):
        curr_t = time.time()

        if not self.running:
            return None

        best_output_data = None
        while True:
            try:
                output_data = self._output_queue.get_nowait()
                if curr_t - output_data[-1] <= max_delay:
                    if (best_output_data is None or
                            output_data[-1] > best_output_data[-1]):
                        best_output_data = output_data
            except queue.Empty:
                break

        return best_output_data

    def thread_shut_down(self, timeout=5.):
        self.running = False

        if self.tracker_thread and self.tracker_thread.is_alive():
            self.tracker_thread.join(timeout=timeout)

        while not self._input_queue.empty():
            try:
                self._input_queue.get_nowait()
            except queue.Empty:
                break

        while not self._output_queue.empty():
            try:
                self._output_queue.get_nowait()
            except queue.Empty:
                break

    def __del__(self):
        self.thread_shut_down()













