"""Stack smoke test: YOLOv8 on one frame, CLIP on one frame. Run before building anything else."""

import os

# torchvision::nms has no MPS kernel, and YOLO runs NMS every frame. Without this the
# predict call raises NotImplementedError on mps. Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import time
import urllib.request
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
TEST_IMG = REPO / "data" / "bus.jpg"
TEST_IMG_URL = "https://ultralytics.com/images/bus.jpg"


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def fetch_test_image() -> Path:
    if not TEST_IMG.exists():
        TEST_IMG.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(TEST_IMG_URL, TEST_IMG)
    return TEST_IMG


def check_yolo(device: str) -> bool:
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    t0 = time.perf_counter()
    results = model.predict(str(TEST_IMG), device=device, verbose=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    boxes = results[0].boxes
    names = results[0].names
    print(f"  device      : {device}")
    print(f"  latency     : {elapsed_ms:.1f} ms (single frame, cold, not a benchmark)")
    print(f"  detections  : {len(boxes)}")
    for cls_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
        print(f"    - {names[int(cls_id)]:<12} conf={conf:.3f}")
    return len(boxes) > 0


def check_clip(device: str) -> bool:
    import open_clip
    from PIL import Image

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device).eval()

    prompts = [
        "a photo of a bus on a city street",
        "a photo of an empty road",
        "a photo of a dog",
        "a photo of people waiting at a bus stop",
    ]

    image = preprocess(Image.open(TEST_IMG)).unsqueeze(0).to(device)
    text = tokenizer(prompts).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        # cosine similarity == dot product once both sides are L2-normalised
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    scores = similarity[0].tolist()
    print(f"  device      : {device}")
    print(f"  latency     : {elapsed_ms:.1f} ms (single frame, cold, not a benchmark)")
    print(f"  embed dim   : {image_features.shape[-1]}")
    for prompt, score in sorted(zip(prompts, scores), key=lambda x: -x[1]):
        print(f"    {score:6.2%}  {prompt}")

    top = max(zip(prompts, scores), key=lambda x: x[1])[0]
    return "bus" in top


def main() -> int:
    device = pick_device()
    print(f"torch {torch.__version__} | MPS available: {torch.backends.mps.is_available()}\n")

    fetch_test_image()

    print("[1/2] YOLOv8n on a single frame")
    yolo_ok = check_yolo(device)
    print(f"  -> {'PASS' if yolo_ok else 'FAIL (no detections)'}\n")

    print("[2/2] CLIP ViT-B/32 zero-shot on a single frame")
    clip_ok = check_clip(device)
    print(f"  -> {'PASS' if clip_ok else 'FAIL (top prompt is not the bus prompt)'}\n")

    ok = yolo_ok and clip_ok
    print("SMOKE TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
