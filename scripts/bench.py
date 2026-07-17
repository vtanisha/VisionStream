"""Benchmark harness. Produces the README table from real measurements.

Design notes that matter for reading the numbers:

* Frames are decoded into RAM up front and cycled. Decode is NOT part of any timing —
  otherwise this would benchmark the video decoder, not the pipeline.
* No source pacing here. The live pipeline paces file playback to the clip's own fps;
  a benchmark must run flat out to find the ceiling.
* detect-only uses model.predict(); +track uses model.track(). Ultralytics fuses
  detection and ByteTrack association inside one call, so tracking cost is only
  observable as the DIFFERENCE between those two configs, not as a separate timer.
* Every config runs in its OWN SUBPROCESS, so no config inherits another's warmed GPU.

* Configs are measured ROUND-ROBIN across repeats, and each row reports the MEDIAN of
  its repeats plus the observed spread. This is not fussiness - it is the only honest
  option on this hardware. A fanless M4 has no thermal steady state, so a config's
  measurement depends on WHEN in the session it ran:

      iteration-based warmup, cold machine:  detect-only rose  28.90 -> 37.35 fps by position
      6s sustained warmup, hot machine:      detect-only fell  36.65 -> 31.26 fps by position

  Same config, ~20-30% swing, driven entirely by run order. Round-robin spreads that
  drift evenly over all configs instead of concentrating it in whichever ran first.
  scripts/order_check.py is the experiment that established this.
"""

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import Settings  # noqa: E402  (sets MPS fallback before torch)

import cv2  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "benchmarks"

# Time-based, not iteration-based. An A/B/A run showed the SAME detect-only config
# measuring 28.90 -> 37.35 fps purely by position in the run order: the M4's GPU ramps
# its clock under sustained load, and a fixed 25-iteration warmup ends long before that
# settles. Warming for a fixed WALL-CLOCK duration puts every config at a comparable
# clock state before the first sample is taken.
WARMUP_SECONDS = 6.0


@dataclass
class Row:
    weights: str
    device: str
    mode: str
    clip_every_n: int | None
    fps: float
    detect_p50: float | None
    detect_p95: float | None
    clip_p50: float | None
    clip_p95: float | None
    encode_p50: float | None
    encode_p95: float | None
    total_p50: float | None
    total_p95: float | None
    frames: int


@dataclass
class Agg:
    """Median across repeats, plus the spread that thermal drift actually produced."""

    weights: str
    device: str
    mode: str
    clip_every_n: int | None
    repeats: int
    fps: float
    fps_min: float
    fps_max: float
    fps_spread_pct: float
    detect_p50: float | None
    detect_p95: float | None
    clip_p50: float | None
    clip_p95: float | None
    encode_p50: float | None
    encode_p95: float | None
    total_p50: float | None
    total_p95: float | None
    frames: int


def _med(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    return round(statistics.median(vals), 2) if vals else None


def aggregate(runs: list[Row]) -> Agg:
    fps_vals = [r.fps for r in runs]
    lo, hi, mid = min(fps_vals), max(fps_vals), statistics.median(fps_vals)
    first = runs[0]
    return Agg(
        weights=first.weights, device=first.device, mode=first.mode,
        clip_every_n=first.clip_every_n, repeats=len(runs),
        fps=round(mid, 2), fps_min=round(lo, 2), fps_max=round(hi, 2),
        fps_spread_pct=round((hi - lo) / mid * 100, 1) if mid else 0.0,
        detect_p50=_med([r.detect_p50 for r in runs]),
        detect_p95=_med([r.detect_p95 for r in runs]),
        clip_p50=_med([r.clip_p50 for r in runs]),
        clip_p95=_med([r.clip_p95 for r in runs]),
        encode_p50=_med([r.encode_p50 for r in runs]),
        encode_p95=_med([r.encode_p95 for r in runs]),
        total_p50=_med([r.total_p50 for r in runs]),
        total_p95=_med([r.total_p95 for r in runs]),
        frames=first.frames,
    )


def pct(samples: list[float], p: float) -> float | None:
    if not samples:
        return None
    ordered = sorted(samples)
    idx = min(int(round(p / 100.0 * len(ordered) + 0.5)) - 1, len(ordered) - 1)
    return round(ordered[max(idx, 0)], 2)


def load_frames(path: Path, limit: int) -> tuple[list, int, int, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while len(frames) < limit:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise SystemExit(f"no frames decoded from {path}")
    return frames, w, h, fps


def run_config(frames, weights, device, mode, clip_every_n, iters) -> Row:
    from ultralytics import YOLO

    settings = Settings()
    settings.device = device
    settings.yolo_weights = weights

    model = YOLO(weights)
    scorer = None
    if mode == "detect+track+clip":
        from backend.pipeline.clip_scorer import ClipScorer

        scorer = ClipScorer(settings)

    from backend.pipeline.draw import draw_overlay
    from backend.pipeline.detector import Track

    det_ms, clip_ms, enc_ms, total_ms = [], [], [], []

    def infer(frame):
        if mode == "detect":
            return model.predict(frame, device=device, verbose=False, conf=settings.conf,
                                 imgsz=settings.imgsz)
        return model.track(frame, persist=True, device=device, verbose=False,
                           conf=settings.conf, imgsz=settings.imgsz, tracker=settings.tracker)

    # Warm up under this config's OWN sustained workload until the clock settles.
    # Covers graph compile, lazy MPS init, and GPU clock ramp.
    w_end = time.perf_counter() + WARMUP_SECONDS
    i = 0
    while time.perf_counter() < w_end:
        infer(frames[i % len(frames)])
        if scorer is not None and i % clip_every_n == 0:
            scorer.score(frames[i % len(frames)])
        i += 1
    if device == "mps":
        torch.mps.synchronize()

    t_start = time.perf_counter()
    for i in range(iters):
        frame = frames[i % len(frames)]
        t0 = time.perf_counter()

        results = infer(frame)
        if device == "mps":
            torch.mps.synchronize()  # else we time the dispatch, not the work
        t1 = time.perf_counter()
        det_ms.append((t1 - t0) * 1000)

        if scorer is not None and i % clip_every_n == 0:
            c0 = time.perf_counter()
            scorer.score(frame)
            if device == "mps":
                torch.mps.synchronize()
            clip_ms.append((time.perf_counter() - c0) * 1000)

        boxes = results[0].boxes
        tracks = []
        if boxes is not None and len(boxes):
            ids = boxes.id.int().tolist() if getattr(boxes, "id", None) is not None \
                else [None] * len(boxes)
            for tid, cid, cf, xy in zip(ids, boxes.cls.int().tolist(), boxes.conf.tolist(),
                                        boxes.xyxy.tolist()):
                tracks.append(Track(tid, cid, model.names[cid], float(cf),
                                    tuple(int(v) for v in xy)))

        e0 = time.perf_counter()
        annotated = draw_overlay(frame, tracks, [], 0.0)
        cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), settings.jpeg_quality])
        e1 = time.perf_counter()
        enc_ms.append((e1 - e0) * 1000)
        total_ms.append((e1 - t0) * 1000)

    wall = time.perf_counter() - t_start

    return Row(
        weights=weights, device=device, mode=mode,
        clip_every_n=clip_every_n if scorer is not None else None,
        fps=round(iters / wall, 2),
        detect_p50=pct(det_ms, 50), detect_p95=pct(det_ms, 95),
        clip_p50=pct(clip_ms, 50), clip_p95=pct(clip_ms, 95),
        encode_p50=pct(enc_ms, 50), encode_p95=pct(enc_ms, 95),
        total_p50=pct(total_ms, 50), total_p95=pct(total_ms, 95),
        frames=iters,
    )


def host_info(clip: Path, w: int, h: int, src_fps: float, n: int) -> dict:
    chip = "unknown"
    try:
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except Exception:
        pass
    return {
        "chip": chip,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "clip_file": clip.name,
        "clip_resolution": f"{w}x{h}",
        "clip_src_fps": round(src_fps, 2),
        "frames_preloaded": n,
        "note": "torchvision::nms has no MPS kernel; it falls back to CPU every frame.",
    }


def to_markdown(rows: list[Agg], meta: dict, sweep: list[Agg]) -> str:
    reps = rows[0].repeats if rows else 0
    lines = [
        "## Benchmark results",
        "",
        f"Measured on **{meta['chip']}** — {meta['platform']}, torch {meta['torch']}, "
        f"Python {meta['python']}.",
        f"Test clip: `{meta['clip_file']}`, **{meta['clip_resolution']}**, source "
        f"{meta['clip_src_fps']} fps, {meta['frames_preloaded']} frames preloaded to RAM "
        "(decode excluded from all timings).",
        f"Each config measured **{reps}× round-robin**; FPS is the median, and *spread* is "
        "(max−min)/median across those repeats.",
        "",
        f"> {meta['note']} So `mps` here means convolutions on GPU, NMS on CPU.",
        "",
        "> **Spread is the honest error bar.** This machine is fanless and has no thermal "
        "steady state, so the same config measures differently depending on when it ran. "
        "Treat any two rows whose spread overlaps as *not distinguishable* by this "
        "benchmark.",
        "",
        "### Throughput and per-stage latency",
        "",
        "| weights | device | mode | FPS (median) | spread | detect p50/p95 (ms) "
        "| CLIP p50/p95 (ms) | encode p50/p95 (ms) | total p50/p95 (ms) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    def fmt(a, b):
        return f"{a} / {b}" if a is not None else "—"

    for r in rows:
        lines.append(
            f"| {r.weights} | {r.device} | {r.mode} | **{r.fps}** | "
            f"±{r.fps_spread_pct}% ({r.fps_min}–{r.fps_max}) | "
            f"{fmt(r.detect_p50, r.detect_p95)} | {fmt(r.clip_p50, r.clip_p95)} | "
            f"{fmt(r.encode_p50, r.encode_p95)} | {fmt(r.total_p50, r.total_p95)} |"
        )

    if sweep:
        sweep = sorted(sweep, key=lambda r: (r.clip_every_n is None, r.clip_every_n or 0))
        lines += [
            "",
            "### CLIP sampling rate vs throughput",
            "",
            "Every Nth frame is scored by CLIP. N=1 scores every frame; *off* is the "
            "detect+track baseline with no CLIP at all.",
            "",
            "| CLIP every N | FPS (median) | spread | CLIP p50 (ms) | total p95 (ms) |",
            "|---:|---:|---:|---:|---:|",
        ]
        for r in sweep:
            label = "off" if r.clip_every_n is None else str(r.clip_every_n)
            lines.append(
                f"| {label} | **{r.fps}** | ±{r.fps_spread_pct}% "
                f"({r.fps_min}–{r.fps_max}) | {r.clip_p50 or '—'} | {r.total_p95} |"
            )

    return "\n".join(lines) + "\n"


def run_isolated(clip: Path, preload: int, weights: str, device: str, mode: str,
                 clip_every_n: int, iters: int) -> Row:
    """Run one config in a fresh interpreter so it cannot inherit another's warm GPU."""
    cmd = [
        sys.executable, str(Path(__file__).resolve()), "--_child",
        "--clip", str(clip), "--preload", str(preload), "--weights", weights,
        "--devices", device, "--mode", mode, "--clip-n", str(clip_every_n),
        "--iters", str(iters),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"child failed for {weights}/{device}/{mode}:\n{proc.stderr[-2000:]}")
    payload = proc.stdout.strip().splitlines()[-1]
    return Row(**json.loads(payload))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", type=Path, default=REPO / "data" / "synthetic_pan_720p.mp4")
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--preload", type=int, default=150)
    ap.add_argument("--devices", default="mps,cpu")
    ap.add_argument("--weights", default="yolov8n.pt,yolov8s.pt")
    ap.add_argument("--sweep", default="1,5,15,30,60")
    ap.add_argument("--quick", action="store_true", help="yolov8n + mps only")
    ap.add_argument("--repeats", type=int, default=3,
                    help="round-robin repeats per config; median is reported")
    ap.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--mode", default="detect", help=argparse.SUPPRESS)
    ap.add_argument("--clip-n", type=int, default=15, help=argparse.SUPPRESS)
    a = ap.parse_args()

    # child: measure exactly one config, emit one JSON line, exit
    if a._child:
        frames, _w, _h, _fps = load_frames(a.clip, a.preload)
        row = run_config(frames, a.weights, a.devices, a.mode,
                         a.clip_n if a.mode.endswith("clip") else None, a.iters)
        print(json.dumps(asdict(row)))
        return 0

    frames, w, h, src_fps = load_frames(a.clip, a.preload)
    meta = host_info(a.clip, w, h, src_fps, len(frames))
    del frames  # children decode their own; don't hold ~400MB while they run

    devices = ["mps"] if a.quick else [d for d in a.devices.split(",") if d]
    weights_list = ["yolov8n.pt"] if a.quick else [x for x in a.weights.split(",") if x]

    # (weights, device, mode, clip_every_n, bucket)
    plan: list[tuple] = []
    for weights in weights_list:
        for device in devices:
            for mode in ("detect", "detect+track", "detect+track+clip"):
                plan.append((weights, device, mode, 15, "main"))
    if not a.quick:
        for n in [int(x) for x in a.sweep.split(",") if x]:
            plan.append(("yolov8n.pt", "mps", "detect+track+clip", n, "sweep"))
        plan.append(("yolov8n.pt", "mps", "detect+track", 15, "sweep"))

    # Round-robin: every config gets sampled at every point on the thermal curve.
    collected: dict[tuple, list[Row]] = {}
    for rep in range(1, a.repeats + 1):
        for weights, device, mode, clip_n, bucket in plan:
            print(f"  [rep {rep}/{a.repeats}] {weights} {device} {mode} n={clip_n} ...",
                  flush=True)
            row = run_isolated(a.clip, a.preload, weights, device, mode, clip_n, a.iters)
            collected.setdefault((weights, device, mode, clip_n, bucket), []).append(row)

    rows = [aggregate(v) for k, v in collected.items() if k[4] == "main"]
    sweep = [aggregate(v) for k, v in collected.items() if k[4] == "sweep"]

    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "rows": [asdict(r) for r in rows], "sweep": [asdict(r) for r in sweep]}
    (RESULTS / "results.json").write_text(json.dumps(payload, indent=2))
    md = to_markdown(rows, meta, sweep)
    (RESULTS / "results.md").write_text(md)
    print("\n" + md)
    print(f"wrote {RESULTS/'results.json'} and {RESULTS/'results.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
