"""Is a config's measurement affected by its POSITION in the run order?

Runs the same config repeatedly, then A/B/A. If FPS climbs with position, the machine
(GPU clock ramp / SoC thermal state) is confounding the benchmark and the results table
cannot be trusted as written.
"""

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "scripts" / "bench.py"
CLIP = REPO / "data" / "synthetic_pan_720p.mp4"


def child(mode: str, iters: int = 80) -> dict:
    cmd = [
        sys.executable, str(BENCH), "--_child", "--clip", str(CLIP), "--preload", "60",
        "--weights", "yolov8n.pt", "--devices", "mps", "--mode", mode,
        "--clip-n", "15", "--iters", str(iters),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stderr[-1500:])
        raise SystemExit("child failed")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> None:
    print("A/B/A: does position change the number?\n")
    plan = [
        ("detect", "A1"),
        ("detect", "A2"),
        ("detect+track+clip", "B1"),
        ("detect", "A3"),
        ("detect+track+clip", "B2"),
        ("detect", "A4"),
    ]
    for mode, tag in plan:
        r = child(mode)
        print(
            f"{tag:>3} {mode:<18} fps={r['fps']:>6.2f}  "
            f"detect_p50={r['detect_p50']:>6.2f}ms  detect_p95={r['detect_p95']:>6.2f}ms"
        )


if __name__ == "__main__":
    main()
