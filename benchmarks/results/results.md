# RF-DETR ONNX Benchmark Results

**Generated:** 2026-02-16 20:04:08

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
| Python | CPU | 1.63 | 68.79 | 0.29 | 70.73 | 14.14 |
| Python | GPU | 1.66 | **8.15** | 0.19 | **9.97** | **100.34** 🚀 |
| C++ | CPU | **0.69** | 56.36 | 0.13 | 57.18 | 17.49 |
| C++ | GPU | 0.72 | 33.49 | **0.08** | 34.33 | 29.13 |

---

## Model: rf-detr-nano.sim

### Images

| Implementation | Device | Preprocess (ms) | ORT Run (ms) | Postprocess (ms) | Total (ms) | FPS |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Python | CPU | 1.80 | 59.35 | 0.30 | 61.42 | 16.28 |
| Python | GPU | 1.71 | 26.35 | 0.18 | 28.22 | 35.44 |
| C++ | CPU | 0.72 | 56.45 | 0.13 | 57.31 | 17.45 |
| C++ | GPU | **0.68** | **7.87** | **0.08** | **8.59** | **116.41** 🚀 |

---

## Model: rf-detr-small

### Images

| Implementation | Device | Preprocess (ms) | ORT Run (ms) | Postprocess (ms) | Total (ms) | FPS |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Python | CPU | 2.69 | 105.72 | 0.29 | 108.71 | 9.20 |
| Python | GPU | 3.88 | 13.07 | 0.18 | 17.15 | 58.32 |
| C++ | CPU | **0.99** | 113.10 | 0.13 | 114.29 | 8.75 |
| C++ | GPU | 1.18 | **11.98** | **0.08** | **13.21** | **75.72** 🚀 |

---

## Model: rf-detr-small.sim

### Images

| Implementation | Device | Preprocess (ms) | ORT Run (ms) | Postprocess (ms) | Total (ms) | FPS |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Python | CPU | 2.71 | 111.83 | 0.29 | 114.79 | 8.71 |
| Python | GPU | 3.90 | 13.22 | 0.20 | 17.34 | 57.67 |
| C++ | CPU | **1.02** | 115.17 | 0.13 | 116.27 | 8.60 |
| C++ | GPU | 1.06 | **12.74** | **0.09** | **13.92** | **71.81** 🚀 |

---

*Note: Best performance values are **highlighted in bold**. For timing metrics (ms), lower is better. For FPS, higher is better.*
