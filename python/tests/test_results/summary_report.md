# RF-DETR Validation Report

## Summary Statistics
- **Average IOU**: 0.9354
- **Average Speedup (ONNX/Torch)**: 56.23x

## Assets Table
| Asset | Torch (ms) | ONNX (ms) | Speedup | Avg IOU | Label Acc | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| drone.jpg | 1137.34 | 20.23 | 56.23x | 0.9354 | 100.0% | ✅ PASS |

## Detailed Results
### drone.jpg
- **Status**: PASS
- **Avg IOU**: 0.9354
- **Speedup**: 56.23x

#### Bounding Box Comparison (Greedy Matching)
We use a greedy matching algorithm (IOU > 0.4) to pair detections from PyTorch and ONNX models.

| Match # | Torch Class | ONNX Class | IOU | Status |
| :--- | :--- | :--- | :---: | :---: |
| 1 | car | car | 0.9909 | ✅ OK |
| 2 | car | car | 0.9830 | ✅ OK |
| 3 | car | car | 0.9586 | ✅ OK |
| 4 | car | car | 0.9945 | ✅ OK |
| 5 | car | car | 0.9899 | ✅ OK |
| 6 | car | car | 0.9920 | ✅ OK |
| 7 | car | car | 0.9838 | ✅ OK |
| 8 | car | car | 0.9932 | ✅ OK |
| 9 | car | car | 0.9707 | ✅ OK |
| 10 | person | person | 0.8614 | ⚠️ DIFF |
| 11 | car | car | 0.9931 | ✅ OK |
| 12 | car | car | 0.9892 | ✅ OK |
| 13 | truck | truck | 0.4419 | ⚠️ DIFF |
| 14 | person | person | 0.9182 | ✅ OK |
| 15 | car | car | 0.9523 | ✅ OK |
| 16 | car | car | 0.9535 | ✅ OK |

| Torch Annotated | ONNX Annotated |
| :---: | :---: |
| ![drone.jpg Torch](drone_torch_annotated.jpg) | ![drone.jpg ONNX](drone_onnx_annotated.jpg) |
