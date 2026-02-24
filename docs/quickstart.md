# Quick Start

This page shows the most common usage patterns to get you running inference quickly.

---

## 1. Prepare a Model

Download a pre-converted ONNX model from [Hugging Face](https://huggingface.co/PierreMarieCurie/rf-detr-onnx/tree/main) and place it in `models/`:

```
models/
└── rf-detr-nano/
    └── rf-detr-nano.sim.onnx
```

Or use the download helper:

```bash
bash tools/download_onnx.sh
```

---

## 2. CLI Inference

Run inference on a single image from the command line:

=== "CPU"

    ```bash
    uv run python python/inference.py \
        --model models/rf-detr-nano/rf-detr-nano.sim.onnx \
        --image assets/drone.jpg \
        --device cpu
    ```

=== "GPU"

    ```bash
    uv run python python/inference.py \
        --model models/rf-detr-nano/rf-detr-nano.sim.onnx \
        --image assets/drone.jpg \
        --device gpu
    ```

Expected output:

```
--- Inference Results ---
Preprocessing:  3.12 ms
ORT Run:        12.45 ms
Postprocessing: 1.87 ms
---------------------------------
Processing (Pre+ORT+Post): 17.44 ms
Processing FPS:           57.34
---------------------------------
Total Latency (inc. I/O):  52.11 ms
Total FPS:                19.19
---------------------------------
Detections saved to: output/output.jpg
```

---

## 3. Python API

### Basic Usage

```python
from modules.model import RFDETRModel

# Initialize (auto-selects best provider: TensorRT > CUDA > CPU)
model = RFDETRModel("models/rf-detr-nano/rf-detr-nano.sim.onnx", device="gpu")

# Run inference from a file path or URL
detections, timings = model.predict("assets/drone.jpg")

# Print results
for det in detections:
    print(f"Label: {det.label}, Score: {det.score:.2f}, Box: {det.unnormalized_box}")
```

### Using an OpenCV Image

```python
import cv2
from modules.model import RFDETRModel

model = RFDETRModel("models/rf-detr-nano/rf-detr-nano.sim.onnx", device="cpu")

image = cv2.imread("assets/drone.jpg")  # BGR format
detections, timings = model.predict(image, confidence_threshold=0.4)

# Save annotated image
model.save_detections(image, detections, "output/result.jpg")
```

### Timing Breakdown

Every `predict()` call returns timing information in milliseconds:

```python
detections, timings = model.predict(image)

print(f"Preprocess:  {timings['preprocess']:.2f} ms")
print(f"ORT Run:     {timings['ort_run']:.2f} ms")
print(f"Postprocess: {timings['postprocess']:.2f} ms")
print(f"Total:       {timings['total']:.2f} ms")
```

---

## 4. C++ Quick Start

After building (see [Installation](installation.md)):

```bash
# Image inference
./rfdetr_image_inference \
    --model ../../models/rf-detr-nano/rf-detr-nano.sim.onnx \
    --image ../../assets/drone.jpg \
    --device gpu

# Video inference
./rfdetr_video_inference \
    --model ../../models/rf-detr-nano/rf-detr-nano.sim.onnx \
    --video ../../assets/sample.mp4 \
    --device gpu
```

---

## 5. Validate Against PyTorch

Verify that the ONNX model produces results matching the original PyTorch model:

```bash
# Run tests using GPU (default)
bash tests/run_tests.sh

# Run tests on CPU
bash tests/run_tests.sh -d cpu

# Pass a custom ONNX Runtime root dir (e.g. when using a manually installed ORT)
bash tests/run_tests.sh -d gpu -o /path/to/onnxruntime
```

---

## Next Steps

- [Python API Reference](python_api.md) — Full documentation of all classes and functions
- [C++ API Reference](cpp_api.md) — C++ library documentation
- [Benchmarks](benchmarks.md) — Measure and compare performance
- [Tools](tools.md) — Export your own fine-tuned model
