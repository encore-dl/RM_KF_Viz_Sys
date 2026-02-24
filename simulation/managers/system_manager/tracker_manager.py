import threading
import queue
import time
import traceback

from simulation.event_bus import event_bus
from simulation.dataflow import Observation, Prediction


class TrackerManager:
    def __init__(self):
        self.running = True

        self._input_queue = queue.Queue(maxsize=10)

        self._tracker = None
        self.tracker_thread = None

        self.target_thread_fps = 300.
        self.fps = 0.
        self.fps_window_len = 10

        self._last_input_timestamp = None

        event_bus.subscribe('obs', self._on_obs)

    def _on_obs(self, data: Observation):
        if not self.running:
            return
        try:
            self._input_queue.put_nowait(data)
        except queue.Full:
            try:
                self._input_queue.get_nowait()
                self._input_queue.put_nowait(data)
            except queue.Empty:
                pass

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
        self._last_input_timestamp = None

        while self.running:
            curr_t = time.time()
            dts.append(curr_t - last_t)
            last_t = curr_t
            if len(dts) == self.fps_window_len:
                self.fps = len(dts) / sum(dts)
                dts.clear()

            thread_start_t = time.time()

            try:
                # 获取输入数据
                input_data = self._input_queue.get(timeout=0.5)

                if self._last_input_timestamp is None:
                    input_dt = 0.01
                else:
                    input_dt = input_data.timestamp - self._last_input_timestamp
                    # input_dt = max(min(input_dt, 0.1), 0.001)
                self._last_input_timestamp = input_data.timestamp

                self._tracker.track(input_data.obs_armors, input_dt, input_data.timestamp)  # 调用track

                # pred = Prediction(
                #     center=self._tracker.pred_pos[0] if self._tracker.pred_pos else None,
                #     armors=self._tracker.pred_pos[1:] if len(self._tracker.pred_pos) > 1 else [],
                #     timestamp=time.time(),
                #     is_tracking=self._tracker.is_tracking,
                #     fps=self.fps,
                #     state_vector=self._tracker.state_vecs
                # )
                # 新增bus数据传递方式，output队列暂留
                # event_bus.publish('pred', pred)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Tracker thread error: {e}")
                print(traceback.format_exc())
                break

            thread_dt = time.time() - thread_start_t
            sleep_t = 1. / self.target_thread_fps - thread_dt
            if sleep_t > 0:
                time.sleep(sleep_t)

    def thread_shut_down(self, timeout=5.):
        self.running = False

        if self.tracker_thread and self.tracker_thread.is_alive():
            self.tracker_thread.join(timeout=timeout)

        while not self._input_queue.empty():
            try:
                self._input_queue.get_nowait()
            except queue.Empty:
                break

    def __del__(self):
        self.thread_shut_down()













