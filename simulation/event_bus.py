import threading
from typing import Dict, List, Callable, Any
from collections import defaultdict


class EventBus:
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._lock = threading.Lock()

    def subscribe(self, topic: str, callback: Callable[[Any], None]):
        # 订阅一个主题，当有消息发布时，callback 会被调用
        with self._lock:
            if topic not in self._subscribers:
                self._subscribers[topic] = []
            self._subscribers[topic].append(callback)

    def publish(self, topic: str, data: Any):
        # 向指定主题发布消息，所有订阅者将收到 data
        # 先获取副本，避免在遍历时修改列表
        with self._lock:
            callbacks = self._subscribers.get(topic, [])[:]
        for cb in callbacks:
            try:
                cb(data)
            except Exception as e:
                print(f"Error in callback for topic '{topic}': {e}")

    # def unsubscribe(self, topic: str):
    #     # 取消指定主题的所有订阅（通常用于清理）
    #     with self._lock:
    #         self._subscribers[topic].clear()


event_bus = EventBus()






