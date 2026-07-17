# VisionStream

Real-time video understanding in the browser. A webcam or uploaded video goes in; YOLOv8
detects objects, ByteTrack gives them persistent IDs across frames, and CLIP scores each
sampled frame zero-shot against a list of text prompts you edit live from the UI. The
annotated stream and the stats render in a Vue 3 SPA.

The point of the prompt editor is that it is the whole "scene understanding" layer. Change
the prompts and the behaviour changes — no retraining, no restart, no new labels.

<!-- TODO(tanisha): drop in a screenshot + a GIF of prompts being edited live.
     The GIF is the money shot for this project — it is the one thing that shows the
     behaviour changing with no retraining. Suggested: start the clip, then swap the
     prompts from street/bus wording to something unrelated and back. -->

## Architecture

```
                 ┌── capture thread ──┐
  webcam / file ─┤                    ├─> [slot: depth 1, drops] ─> detect+track thread
                 └────────────────────┘                                │
                                                                       ├─> draw + JPEG ─> [slot] ─> MJPEG ─> <img> in Vue
                                                                       │
                                              [slot: depth 1] <────────┘ (every Nth frame)
                                                    │
                                                CLIP thread ─> prompt scores ─> /api/stats ─> Vue panel
```

Every hop between threads is a **depth-1 slot that overwrites**, not a queue that buffers.
This is the central design decision. A buffered queue under load keeps FPS looking healthy
while end-to-end latency grows without bound, because the consumer works through a backlog
of stale frames. Dropping instead means a slow stage costs *frames* — which are counted and
reported in the UI — rather than *latency*. Live video has no value in old frames.

CLIP runs on its own thread behind its own slot. Run inline every 15th frame, a ~50 ms CLIP
call would show up as a periodic latency spike in the detection loop. Off-thread it costs
nothing but staleness in the scene label, which is fine: the description of a scene changes
far more slowly than the boxes do.

## How ByteTrack works

ByteTrack turns per-frame detections into tracks with stable IDs. The core insight is in
the name — it keeps the *low* confidence detections instead of throwing them away.

Most trackers filter detections at a threshold (say 0.5) and associate only the survivors.
ByteTrack associates in two passes:

1. **High-confidence pass.** Take the confident detections. For each existing track, predict
   where it should be in this frame with a **Kalman filter** (constant-velocity motion model
   — where was it, how fast was it going, so where is it now). Match predictions to
   detections by **IoU**, solved as a linear assignment problem (this is what the `lap`
   package does — the Hungarian/Jonker-Volgenant solver).
2. **Low-confidence pass.** Tracks that failed to match in pass 1 get a second chance
   against the *low* confidence detections — the ones a normal tracker discards.

Pass 2 is why it survives occlusion. When a person walks behind a pole, the detector doesn't
usually go silent — it emits a weak, partial detection at 0.2 confidence. A threshold-first
tracker drops that, loses the track, and issues a new ID when the person re-emerges (an "ID
switch"). ByteTrack matches the weak detection to the predicted position and keeps the ID.

The reason the weak detections are usable in pass 2 but not pass 1: a low-confidence box on
its own is untrustworthy, but a low-confidence box *that lands where an existing track was
predicted to be* is corroborated by the motion model. Position does the work confidence
can't.

Unmatched tracks aren't deleted immediately; they're kept "lost" for a buffer of frames so a
brief full occlusion doesn't end the track. Unmatched high-confidence detections start new
tracks.

In this repo, Ultralytics runs detection and ByteTrack association inside one `model.track()`
call with `persist=True` — `persist` is what carries tracker state across calls, and it is
the difference between stable IDs and re-detection every frame.

## How CLIP zero-shot scoring works

CLIP was trained on ~400M (image, caption) pairs with a contrastive objective: push an
image's embedding toward its own caption's embedding and away from every other caption's in
the batch. The result is one **shared 512-dimensional space holding both images and text**.

That shared space is the entire trick. Because a photo and a sentence describing it land in
the same neighbourhood, you can compare an image the model has never seen to a sentence
nobody wrote during training, just by measuring the angle between their vectors:

1. Encode each prompt once, L2-normalise it. (Cached here — text embeddings depend only on
   the prompts, so re-encoding per frame would waste a full text forward pass.)
2. Encode the frame, L2-normalise it.
3. Both sides are unit vectors, so their **dot product *is* cosine similarity**.
4. Scale by CLIP's learned temperature and softmax across the prompts.

"Zero-shot" means there is no classifier head and no trained label set — the prompt list
*is* the label set, decided at runtime. That's why editing prompts in the UI changes
behaviour instantly.

**What the score is not:** softmax is taken *across the prompts you supplied*, so it is a
ranking among your options, not a calibrated probability that a thing is present. Give it
four prompts about an empty room and one will still win at 80%, in a room full of people.

## Benchmarks

Measured on **Apple M4** — macOS 26.5 arm64, torch 2.10.0, Python 3.11.14. Test clip
`synthetic_pan_720p.mp4`, **1280×720**, 90 frames preloaded to RAM (decode excluded).
Each config measured **3× round-robin**; FPS is the median; *spread* is (max−min)/median.
Regenerate with `python scripts/bench.py`.

> **Read the spread as the error bar.** This machine is fanless and has no thermal steady
> state, so a config measures differently depending on when it ran. Two rows whose spreads
> overlap are **not distinguishable** by this benchmark. See the caveat below.

### Throughput and per-stage latency

| weights | device | mode | FPS (median) | spread | detect p50/p95 (ms) | CLIP p50/p95 (ms) | encode p50/p95 (ms) | total p50/p95 (ms) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| yolov8n | mps | detect | **30.9** | ±9.0% | 28.5 / 29.6 | — | 3.5 / 3.7 | 33.2 / 34.2 |
| yolov8n | mps | detect+track | **30.1** | ±4.4% | 29.7 / 30.8 | — | 3.5 / 3.7 | 33.4 / 34.4 |
| yolov8n | mps | detect+track+clip | **38.7** | ±18.5% | 17.1 / 21.8 | 56.2 / 74.4 | 3.5 / 4.2 | 21.2 / 80.8 |
| yolov8n | cpu | detect | **14.3** | ±28.8% | 63.7 / 78.0 | — | 4.6 / 4.9 | 68.3 / 82.9 |
| yolov8n | cpu | detect+track | **12.5** | ±26.8% | 74.2 / 94.2 | — | 4.8 / 5.1 | 79.1 / 99.2 |
| yolov8n | cpu | detect+track+clip | **12.8** | ±17.4% | 62.7 / 71.1 | 134.8 / 163.4 | 4.7 / 5.0 | 67.7 / 202.3 |
| yolov8s | mps | detect | **25.4** | ±6.5% | 34.0 / 39.1 | — | 3.6 / 4.0 | 38.7 / 44.2 |
| yolov8s | mps | detect+track | **24.7** | ±8.5% | 37.1 / 42.3 | — | 3.7 / 4.0 | 40.9 / 46.1 |
| yolov8s | mps | detect+track+clip | **28.5** | ±8.6% | 27.8 / 34.0 | 50.9 / 66.7 | 3.6 / 4.1 | 31.9 / 77.6 |
| yolov8s | cpu | detect | **6.1** | ±13.0% | 156.0 / 186.6 | — | 4.8 / 5.1 | 160.8 / 191.5 |
| yolov8s | cpu | detect+track | **5.3** | ±20.5% | 175.7 / 219.4 | — | 4.8 / 5.1 | 180.5 / 224.5 |
| yolov8s | cpu | detect+track+clip | **5.0** | ±16.0% | 176.9 / 213.6 | 150.8 / 168.2 | 4.8 / 5.2 | 183.6 / 331.6 |

**What the table says:**

- **The ≥25 FPS real-time target is met** on the native MPS demo path. YOLOv8n clears it in
  every mode; YOLOv8s sits right at the line (24.7–28.5) and is the accuracy/speed knob.
- **CPU is 2–5× slower**, and this is the honest proxy for the Docker path (Docker on this
  machine has no MPS, so it runs exactly this CPU code — see the two-modes section). YOLOv8n
  on CPU manages ~13 FPS; YOLOv8s on CPU falls to ~5 and is not real-time. *Docker was not
  itself benchmarked — these are native-CPU numbers standing in for it. The container has
  not been built or measured.*
- **The detect+track+clip rows look faster than detect-only, which is not a mistake** — it's
  the GPU-clock effect. YOLOv8n alone doesn't saturate the M4 GPU; CLIP's sustained load
  holds the clocks high and detection rides that (detect p50 drops from 28.5 ms to 17.1 ms).
  This is why the rows are not additive and why the caveat below matters.

### CLIP sampling rate vs throughput — the interesting curve

Every Nth frame is scored by CLIP (`OT_CLIP_EVERY_N`). N=1 scores every frame; *off* is
detect+track with no CLIP. YOLOv8n on MPS:

| CLIP every N | FPS (median) | spread | CLIP p50 (ms) | total p95 (ms) |
|---:|---:|---:|---:|---:|
| 1 (every frame) | **12.7** | ±4.8% | 53.9 | 86.9 |
| 5 | **33.2** | ±11.7% | 46.1 | 69.3 |
| 15 *(default)* | **41.7** | ±1.6% | 56.0 | 65.1 |
| 30 | **42.1** | ±24.1% | 55.0 | 35.2 |
| 60 | **37.1** | ±4.6% | 64.4 | 35.1 |
| off | **29.9** | ±1.3% | — | 34.6 |

Two things worth defending in an interview:

1. **CLIP every frame (N=1) collapses throughput to 12.7 FPS** — CLIP costs ~54 ms/frame,
   so running it inline every frame makes it, not detection, the bottleneck. This is the
   entire reason for frame-sampling. The knee is around **N=15**, which is the default.
2. **"off" (29.9) is *slower* than N=15–60 (37–42)** — same GPU-clock story as above. With
   CLIP off, YOLOv8n never drives the GPU hard enough to clock up, so it plateaus lower than
   when a periodic CLIP burst keeps the clocks warm. Sampled CLIP is close to free here, and
   at the very high end actually *raises* median FPS while widening its spread.

### Methodology, and one caveat that matters

- Frames are decoded to RAM up front and cycled; **decode is excluded from every timing**,
  otherwise this would benchmark the video decoder.
- The live pipeline paces file playback to the clip's own FPS. The benchmark deliberately
  does not — it runs flat out to find the ceiling.
- Each config runs in its **own subprocess**, and warms up under sustained load for a fixed
  wall-clock duration before the first sample.
- `detect` uses `model.predict()`; `detect+track` uses `model.track()`. Ultralytics fuses
  detection and ByteTrack association into a single call, so **tracking cost is only
  observable as the difference between those two rows** — there is no honest way to put a
  separate timer around it here.
- Latency percentiles come from raw per-frame samples, never from averages.

**The caveat:** on this M4, measured per-frame latency depends strongly on GPU clock state,
which depends on how hard the workload drives the GPU. An A/B/A experiment
(`scripts/order_check.py`) showed the *identical* detect-only config measuring 28.9 → 37.4
FPS purely from its position in the run order, before warm-up was made time-based. It also
showed detection running ~40% *faster* when CLIP is in the pipeline, because YOLOv8n alone
does not saturate the M4 GPU and CLIP's sustained load holds the clocks high. Time-based
warm-up controls the ramp, but the clock-state effect is a real property of the hardware.
Read the rows as "this configuration, warmed" rather than as strictly additive costs.

### What is not measured

- **Capture → browser display.** `end_to_end` measures capture → JPEG bytes ready to send.
  Browser decode and paint are not observable from the server, so they are not included and
  not estimated.
- **Tracking accuracy** (MOTA/IDF1, ID switches). The bundled test clip is a synthetic pan
  over a still image, which has no occlusion, so it cannot exercise ID survival. Timings on
  it are representative (real objects, real resolution, identical per-frame compute);
  tracking *quality* on it is not. Drop a real clip in `data/` for that.

## The two run modes

|                | Native (conda + MPS)      | Docker                    |
|----------------|---------------------------|---------------------------|
| Webcam         | Yes                       | **No**                    |
| Video file     | Yes                       | Yes                       |
| Inference      | MPS (GPU conv, CPU NMS)   | **CPU only**              |
| Real-time      | Target ≥25 FPS            | **No — see table**        |
| Purpose        | The demo path             | The portability path      |

This split is not a preference, it's a constraint. **Docker Desktop on Apple Silicon passes
through neither the Metal GPU nor the host camera.** There's no flag to fix it; the VM has
no Metal device and no `/dev/video*`. So the Docker image pins `OT_DEVICE=cpu`, and the API
rejects a webcam request with an explanation rather than failing obscurely.

Do not read the Docker numbers as this project's performance, and do not read the native
numbers as portable. They measure different things on purpose.

### A note on "MPS"

`torchvision::nms` has no MPS kernel, and YOLO runs NMS after every forward pass. So
`PYTORCH_ENABLE_MPS_FALLBACK=1` is set before torch is imported (in `backend/config.py`),
and NMS silently executes on the CPU. **"MPS" here means convolutions on the GPU and NMS on
the CPU, with a device sync every frame** — not a clean end-to-end GPU path.

## Setup — native (the demo path)

```bash
conda env create -f environment.yml
conda activate objecttracker

# smoke test: YOLO on one frame, CLIP on one frame
python scripts/smoke_test.py

# generate the bundled synthetic test clip (or drop your own .mp4 in data/)
python scripts/make_test_clip.py

uvicorn backend.main:app --reload --port 8000
```

Frontend, in a second terminal:

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173
```

Vite proxies `/api` to `localhost:8000`, so the browser stays same-origin.

## Setup — Docker (the portability path)

```bash
docker compose up --build
# UI:  http://localhost:5173
# API: http://localhost:8000/api/health
```

Put `.mp4` files in `./data` — it's mounted into the backend. Webcam will not work here; see
above for why.

## Benchmarks — reproducing

```bash
python scripts/bench.py                    # full matrix -> benchmarks/results.md
python scripts/bench.py --quick            # yolov8n + mps only, fast
python scripts/order_check.py              # the A/B/A order-effect experiment
```

For the Docker (CPU) row:

```bash
docker compose run --rm backend conda run -n objecttracker python scripts/bench.py --devices cpu
```

## Configuration

| Env var           | Default | Meaning                                        |
|-------------------|---------|------------------------------------------------|
| `OT_DEVICE`       | auto    | `mps` or `cpu`. Auto-picks `mps` when available |
| `OT_CLIP_EVERY_N` | `15`    | Score every Nth frame with CLIP                 |
| `OT_YOLO_WEIGHTS` | `yolov8n.pt` | `yolov8n.pt`, `yolov8s.pt`, …             |
| `OT_IN_DOCKER`    | unset   | Set to `1` to disable webcam and force CPU      |

## Limitations — read this part

**CLIP zero-shot is not an anomaly detector.** It is a similarity ranker over a prompt list
you wrote. It has no concept of normal or abnormal, no threshold, and no memory of what this
scene usually looks like.

- **The softmax score is relative, not absolute.** It distributes across your prompts and
  always sums to 1. There is always a winner, even when every prompt is wrong. A high score
  means "most like this prompt, of the ones offered" — never "this is definitely happening."
- **It is genuinely prompt-sensitive.** "a fight" and "two people fighting" and "a photo of
  a fight" can rank differently on the same frame. Prompt phrasing is doing real work here,
  and there is no principled way to choose it other than trying prompts on your footage.
- **Whole-frame only.** CLIP scores the entire frame, and is not connected to the tracks. It
  cannot say *which* tracked object matches a prompt. Per-object CLIP would mean cropping
  each box and encoding it — a real feature, not built here.
- **Sampled, so it is stale by design.** At N=15 and ~30 FPS the label reflects something up
  to half a second old. Fine for describing a scene, wrong for triggering on an event.
- **CLIP on a video frame is off-distribution.** It was trained on curated web images, not
  motion-blurred 720p webcam stills. Expect it to be less sure than the demos suggest.

**On tracking:** ByteTrack is motion-based, with no appearance model (that's the tradeoff
that makes it fast). Two people who cross paths at similar speed can swap IDs, and someone
who leaves and re-enters gets a new ID — it has no way to know it's the same person.

**On the benchmarks:** the bundled clip is synthetic. The FPS numbers are real; the tracking
quality on it is meaningless. See "What is not measured."

## Repo layout

```
backend/
  config.py            settings + device resolution (sets MPS fallback before torch)
  main.py              FastAPI: MJPEG stream, stats, prompt/class control, upload
  pipeline/
    engine.py          thread orchestration, frame-drop policy
    latest_slot.py     depth-1 overwrite slot — the drop policy itself
    detector.py        YOLOv8 + ByteTrack
    clip_scorer.py     CLIP zero-shot scoring, cached text embeddings
    draw.py            boxes, stable per-ID colours, HUD
    source.py          webcam / file, with playback pacing
    stats.py           rolling FPS + p50/p95
scripts/
  smoke_test.py        YOLO + CLIP on a single frame
  bench.py             the benchmark harness
  order_check.py       A/B/A experiment for the GPU clock-state effect
  make_test_clip.py    synthetic test clip generator
frontend/              Vue 3 + Vite SPA
```
