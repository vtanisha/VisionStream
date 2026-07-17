## Benchmark results

Measured on **Apple M4** — macOS-26.5-arm64-arm-64bit, torch 2.10.0, Python 3.11.14.
Test clip: `synthetic_pan_720p.mp4`, **1280x720**, source 30.0 fps, 90 frames preloaded to RAM (decode excluded from all timings).
Each config measured **3× round-robin**; FPS is the median, and *spread* is (max−min)/median across those repeats.

> torchvision::nms has no MPS kernel; it falls back to CPU every frame. So `mps` here means convolutions on GPU, NMS on CPU.

> **Spread is the honest error bar.** This machine is fanless and has no thermal steady state, so the same config measures differently depending on when it ran. Treat any two rows whose spread overlaps as *not distinguishable* by this benchmark.

### Throughput and per-stage latency

| weights | device | mode | FPS (median) | spread | detect p50/p95 (ms) | CLIP p50/p95 (ms) | encode p50/p95 (ms) | total p50/p95 (ms) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| yolov8n.pt | mps | detect | **30.9** | ±9.0% (30.01–32.78) | 28.51 / 29.56 | — | 3.53 / 3.68 | 33.17 / 34.24 |
| yolov8n.pt | mps | detect+track | **30.06** | ±4.4% (29.92–31.23) | 29.71 / 30.78 | — | 3.53 / 3.66 | 33.36 / 34.38 |
| yolov8n.pt | mps | detect+track+clip | **38.71** | ±18.5% (35.19–42.34) | 17.08 / 21.8 | 56.22 / 74.44 | 3.53 / 4.22 | 21.21 / 80.84 |
| yolov8n.pt | cpu | detect | **14.25** | ±28.8% (10.44–14.55) | 63.67 / 78.02 | — | 4.62 / 4.85 | 68.29 / 82.85 |
| yolov8n.pt | cpu | detect+track | **12.48** | ±26.8% (10.8–14.15) | 74.16 / 94.2 | — | 4.81 / 5.1 | 79.07 / 99.19 |
| yolov8n.pt | cpu | detect+track+clip | **12.79** | ±17.4% (10.62–12.85) | 62.7 / 71.06 | 134.82 / 163.37 | 4.72 / 5.0 | 67.69 / 202.25 |
| yolov8s.pt | mps | detect | **25.38** | ±6.5% (24.3–25.94) | 34.02 / 39.1 | — | 3.62 / 4.01 | 38.7 / 44.15 |
| yolov8s.pt | mps | detect+track | **24.73** | ±8.5% (23.06–25.17) | 37.12 / 42.26 | — | 3.67 / 3.97 | 40.85 / 46.1 |
| yolov8s.pt | mps | detect+track+clip | **28.49** | ±8.6% (26.56–29.02) | 27.81 / 33.97 | 50.89 / 66.65 | 3.59 / 4.08 | 31.86 / 77.57 |
| yolov8s.pt | cpu | detect | **6.09** | ±13.0% (5.34–6.13) | 156.03 / 186.56 | — | 4.76 / 5.09 | 160.78 / 191.48 |
| yolov8s.pt | cpu | detect+track | **5.31** | ±20.5% (4.99–6.08) | 175.72 / 219.37 | — | 4.82 / 5.13 | 180.52 / 224.48 |
| yolov8s.pt | cpu | detect+track+clip | **4.99** | ±16.0% (4.94–5.74) | 176.89 / 213.6 | 150.8 / 168.24 | 4.79 / 5.17 | 183.57 / 331.63 |

### CLIP sampling rate vs throughput

Every Nth frame is scored by CLIP. N=1 scores every frame; *off* is the detect+track baseline with no CLIP at all.

| CLIP every N | FPS (median) | spread | CLIP p50 (ms) | total p95 (ms) |
|---:|---:|---:|---:|---:|
| 1 | **12.73** | ±4.8% (12.35–12.96) | 53.89 | 86.88 |
| 5 | **33.16** | ±11.7% (29.39–33.28) | 46.05 | 69.28 |
| 15 | **41.69** | ±1.6% (41.35–42.02) | 55.99 | 65.06 |
| 30 | **42.05** | ±24.1% (33.55–43.7) | 55.01 | 35.21 |
| 60 | **37.11** | ±4.6% (37.07–38.79) | 64.38 | 35.11 |
| off | **29.93** | ±1.3% (29.59–29.97) | — | 34.62 |
