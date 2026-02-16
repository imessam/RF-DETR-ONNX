# RF-DETR ONNX Benchmark Results

**Generated:** 2026-02-16 20:10:25

## Summary

- **Total benchmarks:** 16
- **Python benchmarks:** 8
- **C++ benchmarks:** 8
- **CPU benchmarks:** 8
- **GPU benchmarks:** 8

- **Iterations per benchmark:** 2

---

## Model: rf-detr-nano

### Images

| Implementation | Device | Preprocess (ms) | ORT Run (ms) | Postprocess (ms) | Total (ms) | FPS |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Python | CPU | 1.92 | 139.14 | 0.32 | 141.38 | 7.07 |
| Python | GPU | 1.64 | **9.29** | 0.20 | **11.13** | **89.86** 🚀 |
| C++ | CPU | **0.74** | 74.13 | 0.13 | 75.00 | 13.33 |
| C++ | GPU | 0.81 | 22.56 | **0.08** | 23.46 | 42.63 |

---

## Model: rf-detr-nano.sim

### Images

| Implementation | Device | Preprocess (ms) | ORT Run (ms) | Postprocess (ms) | Total (ms) | FPS |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Python | CPU | 1.74 | 70.12 | 0.31 | 72.18 | 13.86 |
| Python | GPU | 1.60 | 8.75 | 0.30 | 10.66 | 93.85 |
| C++ | CPU | 1.24 | 108.43 | 0.13 | 109.80 | 9.11 |
| C++ | GPU | **0.79** | **8.26** | **0.10** | **9.14** | **109.37** 🚀 |

---

## Model: rf-detr-small

### Images

| Implementation | Device | Preprocess (ms) | ORT Run (ms) | Postprocess (ms) | Total (ms) | FPS |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Python | CPU | 2.93 | 135.40 | 0.30 | 138.63 | 7.21 |
| Python | GPU | 3.83 | **13.10** | 0.20 | **17.12** | **58.40** 🚀 |
| C++ | CPU | **0.93** | 199.30 | 0.13 | 200.37 | 4.99 |
| C++ | GPU | 1.40 | 38.56 | **0.08** | 40.04 | 24.98 |

---

## Model: rf-detr-small.sim

### Images

| Implementation | Device | Preprocess (ms) | ORT Run (ms) | Postprocess (ms) | Total (ms) | FPS |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Python | CPU | 2.86 | 144.11 | 0.36 | 147.32 | 6.79 |
| Python | GPU | 4.68 | 35.68 | 0.23 | 40.60 | 24.63 |
| C++ | CPU | **1.00** | 155.59 | 0.18 | 156.78 | 6.38 |
| C++ | GPU | 1.07 | **11.71** | **0.10** | **12.88** | **77.67** 🚀 |

---

*Note: Best performance values are **highlighted in bold**. For timing metrics (ms), lower is better. For FPS, higher is better.*
