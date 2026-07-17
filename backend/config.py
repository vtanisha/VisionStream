"""Settings + device selection. Import this before anything that imports torch."""

import os

# torchvision::nms has no MPS kernel and YOLO runs NMS every frame. Must be set before
# torch loads or predict() raises NotImplementedError on mps.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from dataclasses import dataclass, field  # noqa: E402

import torch  # noqa: E402

DEFAULT_PROMPTS = [
    "a busy street with people walking",
    "an empty street",
    "a person riding a bicycle",
    "a traffic jam",
]


def resolve_device(requested: str | None = None) -> str:
    """mps > cpu. OT_DEVICE overrides; Docker on Apple Silicon has no MPS passthrough."""
    choice = requested or os.environ.get("OT_DEVICE")
    if choice:
        return choice
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Settings:
    device: str = field(default_factory=resolve_device)
    yolo_weights: str = os.environ.get("OT_YOLO_WEIGHTS", "yolov8n.pt")
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"

    # CLIP is ~10x slower than YOLOv8n per frame; sampling keeps detection real-time.
    clip_every_n: int = int(os.environ.get("OT_CLIP_EVERY_N", "15"))

    conf: float = 0.35
    imgsz: int = 640
    tracker: str = "bytetrack.yaml"
    jpeg_quality: int = 75

    prompts: list[str] = field(default_factory=lambda: list(DEFAULT_PROMPTS))
    class_filter: list[int] | None = None  # None = all COCO classes

    def is_docker(self) -> bool:
        return os.environ.get("OT_IN_DOCKER") == "1"


SETTINGS = Settings()
