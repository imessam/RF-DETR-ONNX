# RF-DETR ONNX Benchmark Results

**Generated:** 2026-02-22 23:14:20

## Summary

- **Total benchmarks:** 8
- **Python benchmarks:** 4
- **C++ benchmarks:** 4
- **CPU benchmarks:** 4
- **GPU benchmarks:** 4

- **Iterations per benchmark:** 10

---

## Model: rf-detr-nanol.sim

### Images

| Implementation | Device | Preprocess (ms) | ORT Run (ms) | Postprocess (ms) | Total (ms) | FPS |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Python | CPU | 1.73 | 63.52 | 0.31 | 65.49 | 15.27 |
| Python | GPU | 1.55 | **7.91** | 0.17 | 9.68 | 103.28 |
| C++ | CPU | **0.73** | 68.76 | 0.13 | 69.83 | 14.32 |
| C++ | GPU | 0.74 | 8.30 | **0.09** | **9.16** | **109.23** 🚀 |

---

## Model: rf-detr-small.sim

### Images

| Implementation | Device | Preprocess (ms) | ORT Run (ms) | Postprocess (ms) | Total (ms) | FPS |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Python | CPU | 3.00 | 129.19 | 0.32 | 132.37 | 7.55 |
| Python | GPU | 5.08 | 31.03 | 0.27 | 35.87 | 27.88 |
| C++ | CPU | **1.20** | 140.89 | 0.14 | 142.28 | 7.03 |
| C++ | GPU | 1.22 | **11.52** | **0.09** | **12.84** | **77.89** 🚀 |

---

*Note: Best performance values are **highlighted in bold**. For timing metrics (ms), lower is better. For FPS, higher is better.*
