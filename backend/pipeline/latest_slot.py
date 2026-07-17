"""Depth-1 frame slot: latest-frame-wins, drops on overwrite.

A normal Queue(maxsize=N) buffers. Under load that keeps FPS looking healthy while
end-to-end latency grows without bound, because consumers work through stale frames.
This slot holds exactly one frame and overwrites it, so a slow consumer costs frames
(counted, reported) instead of latency.
"""

import threading
from typing import Any


class LatestSlot:
    def __init__(self) -> None:
        self._item: Any = None
        self._seq = -1
        self._cv = threading.Condition()
        self._dropped = 0
        self._closed = False

    def put(self, item: Any, seq: int) -> None:
        with self._cv:
            if self._item is not None:
                self._dropped += 1  # overwrote an unconsumed frame
            self._item = item
            self._seq = seq
            self._cv.notify()

    def get(self, timeout: float | None = 1.0) -> tuple[Any, int] | None:
        """Block for the newest frame. None on timeout or close."""
        with self._cv:
            while self._item is None and not self._closed:
                if not self._cv.wait(timeout=timeout):
                    return None
            if self._closed and self._item is None:
                return None
            item, seq = self._item, self._seq
            self._item = None
            return item, seq

    def peek(self) -> tuple[Any, int] | None:
        with self._cv:
            if self._item is None:
                return None
            return self._item, self._seq

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    @property
    def dropped(self) -> int:
        with self._cv:
            return self._dropped
