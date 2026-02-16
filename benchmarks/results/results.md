# RF-DETR ONNX Benchmark Results

**Generated:** 2026-02-16 20:23:40

## Summary

- **Total benchmarks:** 16
- **Python benchmarks:** 8
- **C++ benchmarks:** 8
- **CPU benchmarks:** 8
- **GPU benchmarks:** 8

- **Iterations per benchmark:** 10

---

## Model: rf-detr-nano

### Images

| Implementation | Device | Preprocess (ms) | ORT Run (ms) | Postprocess (ms) | Total (ms) | FPS |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Python | CPU | 1.64 | 59.28 | 0.30 | 61.28 | 16.32 |
| Python | GPU | 1.64 | **8.17** | 0.19 | 10.08 | 99.22 |
| C++ | CPU | **0.68** | 57.11 | 0.13 | 57.87 | 17.28 |
| C++ | GPU | 0.73 | 8.18 | **0.10** | **8.99** | **111.17** 🚀 |

---

## Model: rf-detr-nano.sim

### Images

| Implementation | Device | Preprocess (ms) | ORT Run (ms) | Postprocess (ms) | Total (ms) | FPS |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Python | CPU | 1.67 | 56.33 | 0.30 | 58.31 | 17.15 |
| Python | GPU | 1.71 | 8.15 | 0.17 | 10.02 | 99.78 |
| C++ | CPU | 0.70 | 56.44 | 0.13 | 57.31 | 17.45 |
| C++ | GPU | **0.69** | **7.99** | **0.10** | **8.76** | **114.13** 🚀 |

---

## Model: rf-detr-small

### Images

| Implementation | Device | Preprocess (ms) | ORT Run (ms) | Postprocess (ms) | Total (ms) | FPS |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Python | CPU | 2.74 | 108.88 | 0.30 | 111.82 | 8.94 |
| Python | GPU | 4.70 | **35.33** | 0.19 | 40.26 | 24.84 |
| C++ | CPU | **1.00** | 112.74 | 0.14 | 113.84 | 8.78 |
| C++ | GPU | 1.02 | 36.32 | **0.09** | **37.44** | **26.71** 🚀 |

---

## Model: rf-detr-small.sim

### Images

| Implementation | Device | Preprocess (ms) | ORT Run (ms) | Postprocess (ms) | Total (ms) | FPS |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Python | CPU | 2.70 | 115.04 | 0.30 | 118.13 | 8.47 |
| Python | GPU | 4.07 | 33.45 | 0.18 | 37.81 | 26.45 |
| C++ | CPU | **1.02** | 121.16 | 0.13 | 122.27 | 8.18 |
| C++ | GPU | 1.04 | **12.37** | **0.09** | **13.38** | **74.76** 🚀 |

---

*Note: Best performance values are **highlighted in bold**. For timing metrics (ms), lower is better. For FPS, higher is better.*
