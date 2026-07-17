"""Frame sources: webcam (native only) and video file."""

import time
from pathlib import Path

import cv2
import numpy as np


class VideoSource:
    """Wraps cv2.VideoCapture. Webcam runs live; files are paced to their own FPS and loop."""

    def __init__(self, spec: str | int, loop: bool = True) -> None:
        self.spec = spec
        self.loop = loop
        self.is_webcam = isinstance(spec, int) or str(spec).isdigit()
        if self.is_webcam:
            self.spec = int(spec)

        self.cap = cv2.VideoCapture(self.spec)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open source: {spec!r}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = src_fps if src_fps and src_fps > 0 else 30.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not self.is_webcam else -1
        self._last_read = 0.0

    def read(self) -> np.ndarray | None:
        # A file decodes far faster than real time. Without pacing, a 30fps clip would be
        # consumed at decode speed and the drop counter would measure the decoder, not the
        # pipeline. Webcam needs no pacing - the device is the clock.
        if not self.is_webcam:
            target = 1.0 / self.fps
            elapsed = time.perf_counter() - self._last_read
            if self._last_read and elapsed < target:
                time.sleep(target - elapsed)
            self._last_read = time.perf_counter()

        ok, frame = self.cap.read()
        if not ok:
            if self.is_webcam or not self.loop:
                return None
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            if not ok:
                return None
        return frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()

    def describe(self) -> dict:
        return {
            "spec": str(self.spec),
            "kind": "webcam" if self.is_webcam else "file",
            "width": self.width,
            "height": self.height,
            "src_fps": round(self.fps, 2),
            "frames": self.frame_count,
        }


def resolve_source(spec: str, data_dir: Path) -> "VideoSource":
    if spec.isdigit():
        return VideoSource(int(spec))
    path = Path(spec)
    if not path.is_absolute():
        path = data_dir / spec
    if not path.exists():
        raise FileNotFoundError(f"no such video: {path}")
    return VideoSource(str(path))
