"""YOLOv8 detection + ByteTrack ID association."""

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

from ..config import Settings


@dataclass
class Track:
    track_id: int | None  # None = detected but not yet confirmed into a track
    cls_id: int
    label: str
    conf: float
    xyxy: tuple[int, int, int, int]


class Detector:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = YOLO(settings.yolo_weights)
        self.names: dict[int, str] = self.model.names
        self._warmed = False

    def warmup(self, shape: tuple[int, int] = (640, 640)) -> None:
        """First call compiles the graph and inits MPS lazily (~2.5s). Do it off the hot path."""
        if self._warmed:
            return
        blank = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
        self.model.track(
            blank, persist=True, verbose=False, device=self.settings.device,
            tracker=self.settings.tracker,
        )
        self.model.predictor.trackers[0].reset()
        self._warmed = True

    def track(self, frame: np.ndarray) -> list[Track]:
        """persist=True carries tracker state across calls — that's what makes IDs stable."""
        results = self.model.track(
            frame,
            persist=True,
            verbose=False,
            device=self.settings.device,
            tracker=self.settings.tracker,
            conf=self.settings.conf,
            imgsz=self.settings.imgsz,
            classes=self.settings.class_filter,
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        # .id is None for detections ByteTrack has not yet promoted to a confirmed track
        ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)
        out: list[Track] = []
        for tid, cls_id, conf, xyxy in zip(
            ids, boxes.cls.int().tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()
        ):
            out.append(
                Track(
                    track_id=tid,
                    cls_id=cls_id,
                    label=self.names[cls_id],
                    conf=float(conf),
                    xyxy=tuple(int(v) for v in xyxy),
                )
            )
        return out

    def reset_tracker(self) -> None:
        """Call on source switch, or IDs leak across videos."""
        if self.model.predictor is not None and getattr(self.model.predictor, "trackers", None):
            for tracker in self.model.predictor.trackers:
                tracker.reset()
