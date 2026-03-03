# RF-DETR ONNX Benchmark Results

**Generated:** 2026-03-03 20:50:45

## System Information

| Component | Details |
| :--- | :--- |
| **CPU** | AMD Ryzen 5 7600 6-Core Processor |
| **CPU Cores / Threads** | 6 cores / 12 threads |
| **RAM** | 7.3 GB |
| **GPU** | NVIDIA GeForce RTX 4060 (8.0 GB VRAM, driver 591.74) |

---

## Summary

- **Total benchmarks:** 20
- **Python benchmarks:** 10
- **C++ benchmarks:** 10
- **CPU benchmarks:** 10
- **GPU benchmarks:** 10

- **Iterations per benchmark:** 10

## Model: rf-detr-nano.sim

### Images

| Implementation | Device | Preprocess (ms) | ORT Run (ms) | Postprocess (ms) | Total (ms) | FPS |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Python | CPU | 1.75 | 56.79 | 0.30 | 58.81 | 17.01 |
| Python | GPU | 1.68 | **8.23** | 0.16 | 10.04 | 99.64 |
| C++ | CPU | **0.79** | 87.95 | 0.03 | 88.75 | 11.27 |
| C++ | GPU | 0.93 | 8.39 | **0.03** | **9.36** | **106.88** 🚀 |

---

## Model: rf-detr-nano.sim_fp16

### Images

| Implementation | Device | Preprocess (ms) | ORT Run (ms) | Postprocess (ms) | Total (ms) | FPS |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Python | CPU | 1.99 | 141.82 | 0.80 | 144.57 | 6.92 |
| Python | GPU | 1.89 | 12.39 | 0.77 | 15.20 | 65.77 |
| C++ | CPU | 1.01 | 170.87 | 0.04 | 171.83 | 5.82 |
| C++ | GPU | **0.99** | **9.56** | **0.04** | **10.57** | **94.58** 🚀 |

---

## Model: rf-detr-nano.sim_keep_io_types_fp16

### Images

| Implementation | Device | Preprocess (ms) | ORT Run (ms) | Postprocess (ms) | Total (ms) | FPS |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Python | CPU | 2.26 | 121.79 | 0.34 | 124.78 | 8.01 |
| Python | GPU | 1.64 | **8.54** | 0.25 | 10.48 | 95.46 |
| C++ | CPU | **0.82** | 143.46 | **0.03** | 144.41 | 6.92 |
| C++ | GPU | 0.89 | 9.20 | 0.03 | **10.05** | **99.54** 🚀 |

---

## Model: rf-detr-small.sim

### Images

| Implementation | Device | Preprocess (ms) | ORT Run (ms) | Postprocess (ms) | Total (ms) | FPS |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Python | CPU | 2.72 | 112.10 | 0.30 | 114.95 | 8.70 |
| Python | GPU | 4.14 | **32.22** | 0.18 | 36.30 | 27.55 |
| C++ | CPU | 1.20 | 168.97 | 0.03 | 171.09 | 5.84 |
| C++ | GPU | **1.00** | 32.67 | **0.03** | **33.59** | **29.77** 🚀 |

---

## Model: rf-detr-small.sim_fp16

### Images

| Implementation | Device | Preprocess (ms) | ORT Run (ms) | Postprocess (ms) | Total (ms) | FPS |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Python | CPU | 2.76 | 170.52 | 0.76 | 174.09 | 5.74 |
| Python | GPU | 3.73 | 32.19 | 0.70 | 36.44 | 27.44 |
| C++ | CPU | 1.29 | 331.91 | 0.04 | 333.02 | 3.00 |
| C++ | GPU | **0.94** | **11.64** | **0.04** | **12.61** | **79.30** 🚀 |

---

*Note: Best performance values are **highlighted in bold**. For timing metrics (ms), lower is better. For FPS, higher is better.*
