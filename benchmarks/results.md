# RF-DETR Benchmark Results

Generated on: 2026-02-16 01:06:26

## Test Case: Multi Images

| Implementation | Device | Prepro (ms) | ORT (ms) | Post (ms) | Total (ms) | FPS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Python | CPU | 3.76 | 61.68 | 0.34 | 65.77 | 15.20 |
| Python | GPU | 2.14 | 15.83 | 0.21 | 18.17 | 55.03 |
| C++ | CPU | 1.54 | 60.78 | 0.14 | 62.46 | 16.01 |
| C++ | GPU | **0.81** | **15.75** | **0.13** | **16.68** | **59.94** 🚀 |

## Test Case: Video

| Implementation | Device | Prepro (ms) | ORT (ms) | Post (ms) | Total (ms) | FPS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Python | CPU | 3.93 | 59.72 | 0.43 | 64.07 | 15.61 |
| Python | GPU | 2.14 | 15.83 | 0.21 | 18.18 | 55.00 |
| C++ | CPU | 2.15 | 104.16 | 0.14 | 106.45 | 9.39 |
| C++ | GPU | **0.81** | **15.75** | **0.12** | **16.69** | **59.91** 🚀 |

