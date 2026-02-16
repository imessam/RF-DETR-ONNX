# Accuracy Validation Report

| Asset | Implementation | Torch (ms) | ONNX (ms) | Speedup | Avg IOU | Status |
| :--- | :--- | ---: | ---: | ---: | ---: | :---: |
| coco_1 | Python ONNX | 1158.75 | 21.88 | 53.0x | 0.9649 | ✅ PASS |
| | C++ ONNX | 1158.75 | 21.72 | 53.3x | 0.9648 | ✅ PASS |
| | | | | | | |