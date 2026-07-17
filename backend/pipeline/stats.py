"""Rolling throughput + latency percentiles. Percentiles come from raw samples, never averaged."""

import threading
import time
from collections import deque


class Rolling:
    """Fixed-window sample buffer with p50/p95 readout."""

    def __init__(self, window: int = 300) -> None:
        self._samples: deque[float] = deque(maxlen=window)
        self._lock = threading.Lock()

    def add(self, value: float) -> None:
        with self._lock:
            self._samples.append(value)

    def percentile(self, pct: float) -> float | None:
        with self._lock:
            if not self._samples:
                return None
            ordered = sorted(self._samples)
        # nearest-rank; on a 300-sample window the interpolation difference is noise
        idx = min(int(round(pct / 100.0 * len(ordered) + 0.5)) - 1, len(ordered) - 1)
        return ordered[max(idx, 0)]

    def mean(self) -> float | None:
        with self._lock:
            if not self._samples:
                return None
            return sum(self._samples) / len(self._samples)

    def clear(self) -> None:
        with self._lock:
            self._samples.clear()


class StageTimers:
    """Per-stage latency in ms, broken out for the benchmark table."""

    # No "track" stage on purpose: ultralytics runs detection and ByteTrack association
    # inside one model.track() call, so there is nothing to put a timer around. Tracking
    # cost shows up only as the detect vs detect+track delta in scripts/bench.py.
    STAGES = ("detect", "clip", "encode", "end_to_end")

    def __init__(self, window: int = 300) -> None:
        self.stages = {name: Rolling(window) for name in self.STAGES}
        self._fps_marks: deque[float] = deque(maxlen=120)
        self._lock = threading.Lock()

    def record(self, stage: str, ms: float) -> None:
        self.stages[stage].add(ms)

    def mark_frame(self) -> None:
        with self._lock:
            self._fps_marks.append(time.perf_counter())

    def fps(self) -> float:
        """Measured over the marks actually in the window, not a nominal rate."""
        with self._lock:
            if len(self._fps_marks) < 2:
                return 0.0
            span = self._fps_marks[-1] - self._fps_marks[0]
            if span <= 0:
                return 0.0
            return (len(self._fps_marks) - 1) / span

    def snapshot(self) -> dict:
        out = {"fps": round(self.fps(), 2)}
        for name, roll in self.stages.items():
            p50, p95 = roll.percentile(50), roll.percentile(95)
            out[name] = {
                "p50_ms": round(p50, 2) if p50 is not None else None,
                "p95_ms": round(p95, 2) if p95 is not None else None,
            }
        return out

    def reset(self) -> None:
        for roll in self.stages.values():
            roll.clear()
        with self._lock:
            self._fps_marks.clear()
