"""Pipeline orchestration.

  capture thread -> [slot] -> infer thread -> [broadcast slot] -> MJPEG response
                       \-> [clip slot] -> clip thread

Every hop is a depth-1 LatestSlot, so a slow stage drops frames instead of growing a
backlog. CLIP sits on its own thread behind its own slot: a ~40ms CLIP call on every
15th frame, run inline, would show up as a periodic latency spike in the detect loop.
Off-thread, it costs nothing but staleness in the label - which is what we want, since
the scene description changes far slower than the boxes do.
"""

import threading
import time

import cv2
import numpy as np

from ..config import Settings
from .clip_scorer import ClipScorer
from .detector import Detector, Track
from .draw import draw_overlay
from .latest_slot import LatestSlot
from .source import VideoSource
from .stats import StageTimers


class Engine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.detector = Detector(settings)
        self.clip = ClipScorer(settings)
        self.timers = StageTimers()

        self._source: VideoSource | None = None
        self._capture_slot = LatestSlot()
        self._clip_slot = LatestSlot()
        self._jpeg: bytes | None = None
        self._jpeg_cv = threading.Condition()

        self._threads: list[threading.Thread] = []
        self._running = threading.Event()
        self._lock = threading.Lock()

        self._seq = 0
        self._unique_ids: set[int] = set()
        self._current_tracks: list[Track] = []
        self._clip_result: list[tuple[str, float]] = []
        self._clip_frames_scored = 0
        self._source_info: dict = {}

    # ---- lifecycle -------------------------------------------------------

    def warmup(self) -> None:
        self.detector.warmup()
        self.clip.warmup()

    def start(self, source: VideoSource) -> None:
        self.stop()
        self.detector.reset_tracker()  # else IDs carry over from the previous video
        with self._lock:
            self._source = source
            self._source_info = source.describe()
            self._unique_ids.clear()
            self._current_tracks = []
            self._clip_result = []
            self._clip_frames_scored = 0
            self._seq = 0
        self.timers.reset()
        self._capture_slot = LatestSlot()
        self._clip_slot = LatestSlot()

        self._running.set()
        self._threads = [
            threading.Thread(target=self._capture_loop, name="capture", daemon=True),
            threading.Thread(target=self._infer_loop, name="infer", daemon=True),
            threading.Thread(target=self._clip_loop, name="clip", daemon=True),
        ]
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        self._running.clear()
        self._capture_slot.close()
        self._clip_slot.close()
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads = []
        with self._lock:
            if self._source is not None:
                self._source.release()
                self._source = None

    @property
    def running(self) -> bool:
        return self._running.is_set()

    # ---- threads ---------------------------------------------------------

    def _capture_loop(self) -> None:
        while self._running.is_set():
            src = self._source
            if src is None:
                break
            frame = src.read()
            if frame is None:
                self._running.clear()
                break
            self._seq += 1
            # capture_ts anchors end-to-end latency to the moment the frame existed
            self._capture_slot.put((frame, time.perf_counter()), self._seq)

    def _infer_loop(self) -> None:
        while self._running.is_set():
            got = self._capture_slot.get(timeout=0.5)
            if got is None:
                continue
            (frame, capture_ts), seq = got

            t0 = time.perf_counter()
            tracks = self.detector.track(frame)
            t_detect = time.perf_counter()
            # ultralytics fuses detection and ByteTrack association in one call, so the
            # split below is honest about what it can and cannot separate.
            self.timers.record("detect", (t_detect - t0) * 1000)

            with self._lock:
                self._current_tracks = tracks
                for tr in tracks:
                    if tr.track_id is not None:
                        self._unique_ids.add(tr.track_id)
                clip_result = list(self._clip_result)

            if seq % self.settings.clip_every_n == 0:
                self._clip_slot.put((frame.copy(), time.perf_counter()), seq)

            t_enc0 = time.perf_counter()
            annotated = draw_overlay(frame, tracks, clip_result, self.timers.fps())
            ok, buf = cv2.imencode(
                ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), self.settings.jpeg_quality]
            )
            t_enc1 = time.perf_counter()
            if not ok:
                continue
            self.timers.record("encode", (t_enc1 - t_enc0) * 1000)

            # capture -> encoded bytes ready to send. Browser decode+paint is NOT included;
            # that is not observable from the server. README says so.
            self.timers.record("end_to_end", (t_enc1 - capture_ts) * 1000)
            self.timers.mark_frame()

            with self._jpeg_cv:
                self._jpeg = buf.tobytes()
                self._jpeg_cv.notify_all()

    def _clip_loop(self) -> None:
        while self._running.is_set():
            got = self._clip_slot.get(timeout=0.5)
            if got is None:
                continue
            (frame, _ts), _seq = got
            t0 = time.perf_counter()
            try:
                result = self.clip.score(frame)
            except Exception:
                continue
            self.timers.record("clip", (time.perf_counter() - t0) * 1000)
            with self._lock:
                self._clip_result = result
                self._clip_frames_scored += 1

    # ---- consumers -------------------------------------------------------

    def jpeg_stream(self):
        """Yields multipart MJPEG chunks. Waits for new frames rather than re-sending stale ones."""
        last = None
        while True:
            with self._jpeg_cv:
                if not self._jpeg_cv.wait_for(
                    lambda: self._jpeg is not None and self._jpeg is not last, timeout=1.0
                ):
                    if not self._running.is_set():
                        break
                    continue
                frame = self._jpeg
                last = frame
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n"
            )

    def stats(self) -> dict:
        with self._lock:
            tracks = list(self._current_tracks)
            unique = len(self._unique_ids)
            clip_result = list(self._clip_result)
            scored = self._clip_frames_scored
            info = dict(self._source_info)

        top = clip_result[0] if clip_result else None
        counts: dict[str, int] = {}
        for tr in tracks:
            counts[tr.label] = counts.get(tr.label, 0) + 1

        return {
            "running": self._running.is_set(),
            "device": self.settings.device,
            "source": info,
            "objects_tracked": len(tracks),
            "unique_ids": unique,
            "class_counts": counts,
            "top_prompt": {"prompt": top[0], "score": round(top[1], 4)} if top else None,
            "clip_scores": [{"prompt": p, "score": round(s, 4)} for p, s in clip_result],
            "clip_every_n": self.settings.clip_every_n,
            "clip_frames_scored": scored,
            "frames_dropped_capture": self._capture_slot.dropped,
            "frames_dropped_clip": self._clip_slot.dropped,
            "timings": self.timers.snapshot(),
            "tracks": [
                {"id": t.track_id, "label": t.label, "conf": round(t.conf, 3)} for t in tracks
            ],
        }
