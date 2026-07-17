"""Generate a test clip by panning a viewport across bus.jpg.

Stand-in only. Compute per frame is representative (real objects, real resolution), so
FPS/latency measured on it are meaningful. Motion is synthetic and occlusion-free, so
tracking QUALITY (ID switches, occlusion survival) is NOT exercised. Use a real clip
for that.
"""

import argparse
import math
import urllib.request
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
SRC_IMG = REPO / "data" / "bus.jpg"
SRC_URL = "https://ultralytics.com/images/bus.jpg"


def build(out: Path, seconds: int, fps: int, width: int, height: int) -> None:
    if not SRC_IMG.exists():
        SRC_IMG.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SRC_URL, SRC_IMG)

    img = cv2.imread(str(SRC_IMG))
    # bus.jpg is portrait (810x1080). Fit it to the viewport WIDTH so the whole scene is
    # visible, then pan vertically down the remaining height. Scaling to 2x width instead
    # crops to a tiny region: the viewport lands on one door of the bus, YOLO sees 2
    # objects instead of 6, and the clip stops being representative.
    canvas = cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])))
    ch, cw = canvas.shape[:2]
    if ch < height:  # too short to pan - letterbox rather than crash
        pad = np.zeros((height - ch, cw, 3), dtype=np.uint8)
        canvas = np.vstack([canvas, pad])
        ch = canvas.shape[0]

    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    total = seconds * fps

    for i in range(total):
        phase = i / total
        # ease in/out vertical pan so objects enter and leave the viewport
        y = int((ch - height) * (0.5 - 0.5 * math.cos(2 * math.pi * phase)))
        y = max(0, min(y, ch - height))
        writer.write(canvas[y : y + height, 0:width])

    writer.release()
    print(f"wrote {out}  {width}x{height}  {seconds}s @ {fps}fps  ({total} frames)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "data" / "synthetic_pan_720p.mp4")
    ap.add_argument("--seconds", type=int, default=20)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    a = ap.parse_args()
    build(a.out, a.seconds, a.fps, a.width, a.height)
