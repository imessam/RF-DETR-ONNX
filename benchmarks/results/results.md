# RF-DETR ONNX Benchmark Results

**Generated:** 2026-03-01 22:14:55

## System Information

| Component | Details |
| :--- | :--- |
| **CPU** | AMD Ryzen 5 7600 6-Core Processor |
| **CPU Cores / Threads** | 6 cores / 12 threads |
| **RAM** | 7.3 GB |
| **GPU** | NVIDIA GeForce RTX 4060 (8.0 GB VRAM, driver 591.74) |

---

## Summary

- **Total benchmarks:** 8
- **Python benchmarks:** 4
- **C++ benchmarks:** 4
- **CPU benchmarks:** 4
- **GPU benchmarks:** 4

- **Iterations per benchmark:** 10

## Model: rf-detr-nanol.sim

### Images

| Implementation | Device | Preprocess (ms) | ORT Run (ms) | Postprocess (ms) | Total (ms) | FPS |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Python | CPU | 2.21 | 67.31 | 0.37 | 69.86 | 14.31 |
| Python | GPU | 1.65 | 7.83 | 0.17 | 9.62 | 103.93 |
| C++ | CPU | 0.77 | 56.15 | 0.13 | 58.17 | 17.19 |
| C++ | GPU | **0.61** | **7.82** | **0.09** | **8.47** | **118.01** 🚀 |

---

## Model: rf-detr-small.sim

### Images

| Implementation | Device | Preprocess (ms) | ORT Run (ms) | Postprocess (ms) | Total (ms) | FPS |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Python | CPU | 2.77 | 117.45 | 0.29 | 120.53 | 8.30 |
| Python | GPU | 4.06 | 31.79 | 0.18 | 36.17 | 27.65 |
| C++ | CPU | 0.98 | 119.17 | 0.13 | 120.22 | 8.32 |
| C++ | GPU | **0.94** | **12.79** | **0.09** | **13.74** | **72.80** 🚀 |

---

*Note: Best performance values are **highlighted in bold**. For timing metrics (ms), lower is better. For FPS, higher is better.*
