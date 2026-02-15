# RF-DETR Benchmark Results

Generated on: 2026-02-16 00:20:02

## Test Case: Multi Images

| Implementation | Device | Prepro (ms) | ORT (ms) | Post (ms) | Total (ms) | FPS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Python | CPU | 6.73 | 96.93 | 0.77 | 104.43 | 9.58 |
| Python | GPU | 2.13 | **15.80** | **0.21** | 18.14 | 55.12 |
| C++ | CPU | 4.87 | 103.05 | 0.76 | 108.69 | 9.20 |
| C++ | GPU | **0.57** | 15.83 | 0.66 | **17.07** | **58.59** 🚀 |

## Test Case: Video

| Implementation | Device | Prepro (ms) | ORT (ms) | Post (ms) | Total (ms) | FPS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Python | CPU | 10.65 | 64.03 | 0.94 | 75.62 | 13.22 |
| Python | GPU | 2.57 | 15.92 | **0.23** | 18.71 | 53.44 |
| C++ | CPU | 5.53 | 81.07 | 1.17 | 87.77 | 11.39 |
| C++ | GPU | **0.54** | **15.81** | 0.66 | **17.01** | **58.78** 🚀 |

