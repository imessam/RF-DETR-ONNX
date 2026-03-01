# RF-DETR with ONNX

[![CI](https://github.com/imessam/RF-DETR-ONNX/actions/workflows/ci.yml/badge.svg)](https://github.com/imessam/RF-DETR-ONNX/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Models-yellow)](https://huggingface.co/PierreMarieCurie/rf-detr-onnx/tree/main)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://imessam.github.io/RF-DETR-ONNX/)

A modular, production-ready library for running **RF-DETR** object detection and instance segmentation inference with **ONNX Runtime** — in both Python and C++.

This repository is a fork of the original work by [PierreMarieCurie](https://github.com/PierreMarieCurie/rf-detr-onnx), reworked into a modular structure with additional features like manual device selection, performance metrics, and a high-performance C++ library. Special thanks to [PierreMarieCurie](https://github.com/PierreMarieCurie) for the initial implementation and model conversions.

RF-DETR is a transformer-based object detection and instance segmentation architecture developed by Roboflow. For more details, see the [RF-DETR repository](https://github.com/roboflow/rf-detr/tree/main).

| Original Image | Torch Reference | ONNX Inference Result |
|----------------|-----------------|-----------------------|
| <p align="center"><img src="assets/drone.jpg" width="100%"></p> | <p align="center"><img src="assets/reference_demo.jpg" width="100%"></p> | <p align="center"><img src="assets/detection_demo.jpg" width="100%"></p> |

---

## Quick Start

```bash
git clone https://github.com/imessam/rf-detr-onnx.git
cd rf-detr-onnx
uv sync
uv run python python/inference.py --model models/rf-detr-nano/rf-detr-nano.sim.onnx --image assets/drone.jpg
```

Or, using Python directly:

```python
from modules.model import RFDETRModel

model = RFDETRModel("models/rf-detr-nano/rf-detr-nano.sim.onnx", device="cpu")
detections, timings = model.predict("assets/drone.jpg")
model.save_detections("assets/drone.jpg", detections, "output/result.jpg")
```

---

## Project Structure

```
rf-detr-onnx/
├── python/             # Python implementation
│   ├── inference.py    # CLI inference demo
│   └── modules/        # Core library (model, session, utils)
├── cpp/                # High-performance C++ implementation
│   ├── include/        # Header files
│   ├── src/            # Source code
│   └── CMakeLists.txt  # Build configuration
├── benchmarks/         # Automated performance benchmarking suite
├── tools/              # Model export and conversion scripts
├── tests/              # Validation and quality assurance
└── assets/             # Demo images
```

### Components

| Component | Description |
|-----------|-------------|
| [python/](python/) | Modular Python library for fast prototyping and high-level use |
| [cpp/](cpp/) | High-performance C++ library for production deployment |
| [benchmarks/](benchmarks/) | Automated benchmarking suite (Python & C++, CPU & GPU) |
| [tools/](tools/) | Export scripts for converting RF-DETR checkpoints to ONNX |
| [tests/](tests/) | Accuracy validation pipeline against PyTorch reference |

---

## Installation

### Using uv (Recommended)

[uv](https://docs.astral.sh/uv/) is the recommended package manager. Install it first, then:

```bash
git clone https://github.com/imessam/rf-detr-onnx.git
cd rf-detr-onnx

# GPU acceleration is the default (onnxruntime-gpu is a base dependency)
uv sync

# With model export support
uv sync --extra export

# With test support
uv sync --extra test

# With documentation tools
uv sync --extra docs
```

> **CPU-only:** `onnxruntime-gpu` is installed by default via `uv sync`. If you don't have a GPU,
> override it after syncing:
> ```bash
> uv sync
> .venv/bin/pip uninstall -y onnxruntime-gpu
> .venv/bin/pip install onnxruntime
> ```

### Using pip

```bash
pip install .

# With optional extras
pip install ".[export]"   # for model export
pip install ".[test]"     # for running tests
```

> **CPU-only:** If you don't have a GPU, replace `onnxruntime-gpu` with `onnxruntime`:
> ```bash
> pip uninstall -y onnxruntime-gpu && pip install onnxruntime
> ```

---

## Inference

### CLI

```bash
# CPU inference
uv run python python/inference.py \
    --model models/rf-detr-nano/rf-detr-nano.sim.onnx \
    --image assets/drone.jpg \
    --device cpu

# GPU inference
uv run python python/inference.py \
    --model models/rf-detr-nano/rf-detr-nano.sim.onnx \
    --image assets/drone.jpg \
    --device gpu
```

#### CLI Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | **Required** | Path to the `.onnx` model file |
| `--image` | **Required** | Path or URL to the input image |
| `--output` | `output/output.jpg` | Path to save the annotated output image |
| `--threshold` | `0.5` | Confidence threshold for filtering detections |
| `--max_number_boxes` | `300` | Maximum number of boxes to return |
| `--device` | `gpu` | Device: `gpu` (tries TensorRT → CUDA → CPU) or `cpu` |

### Python API

```python
from modules.model import RFDETRModel

# Initialize (prefers TensorRT > CUDA > CPU automatically)
model = RFDETRModel("path/to/model.onnx", device="gpu")

# Run inference on a file path
detections, timings = model.predict("path/to/image.jpg")

# Or on an OpenCV/NumPy image (BGR)
import cv2
image = cv2.imread("path/to/image.jpg")
detections, timings = model.predict(image, confidence_threshold=0.4)

# Visualize and save
model.save_detections(image, detections, "output/result.jpg")

# Inspect timing breakdown
print(f"Preprocess:  {timings['preprocess']:.2f} ms")
print(f"ORT Run:     {timings['ort_run']:.2f} ms")
print(f"Postprocess: {timings['postprocess']:.2f} ms")
print(f"Total:       {timings['total']:.2f} ms")
```

---

## C++ Implementation

The C++ library provides high-performance inference for production deployment.

### Prerequisites

Requires: OpenCV 4.x, ONNX Runtime C++ API, and CMake ≥ 3.15.

### Build

```bash
cd cpp
mkdir build && cd build
cmake .. -DONNXRUNTIME_ROOT_DIR=/path/to/onnxruntime
make -j$(nproc)
```

> **Note:** Set `ONNXRUNTIME_ROOT_DIR` to the root of your ONNX Runtime C++ installation (contains `include/` and `lib/`). Defaults to `/opt/onnxruntime`.

#### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `ENABLE_EXAMPLES` | `ON` | Build example executables |
| `ENABLE_BENCHMARKS` | `ON` | Build benchmark executables |
| `ENABLE_TESTS` | `OFF` | Build test executables |

```bash
cmake .. \
  -DENABLE_EXAMPLES=ON \
  -DENABLE_BENCHMARKS=ON \
  -DENABLE_TESTS=OFF
```

### Run

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

### Install as Library

```bash
cd cpp && mkdir build && cd build
cmake .. -DONNXRUNTIME_ROOT_DIR=/path/to/onnxruntime
make -j$(nproc)
cmake --install .
```

This installs `librfdetr_onnx.a` and headers into your CMake install prefix. Consume via CMake:

```cmake
find_package(rfdetr_onnx REQUIRED)
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE rfdetr_onnx::rfdetr_onnx)
```

---

## Validation & Testing

A fully automated pipeline verifies that the exported ONNX model matches the original PyTorch model's accuracy.

```bash
# Run full validation pipeline (GPU)
bash tests/run_tests.sh -d gpu

# Run on CPU
bash tests/run_tests.sh -d cpu
```

### Export Your Own Model

```bash
uv sync --extra export
uv run python tools/export_roboflow.py \
    --weights path/to/rf-detr-nano.pth \
    --model-type nano
```

#### Export Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--weights` | **Required** | Path to the `.pth` checkpoint file |
| `--model-type` | **Required** | Architecture: `nano`, `small`, `base`, `medium`, `large` |
| `--output-dir` | `models/` | Directory to save the exported model (output placed in `<dir>/<checkpoint-stem>/`) |
| `--opset` | `17` | ONNX opset version |
| `--no-simplify` | `False` | Disable model simplification |

---

## Benchmarking

The benchmarking suite evaluates both Python and C++ implementations across CPU and GPU providers.

```bash
cd benchmarks
./run_benchmarks.sh -n 10 -v
```

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `-n <int>` | `10` | Number of benchmark iterations. |
| `-c <float>` | `2.0` | Cooldown period (seconds) between runs. |
| `-s <float>` | `0.1` | Sleep delay between per-image iterations. |
| `-v` | `Off` | Enable verbose per-iteration logging. |
| `-u <url>` | `https://github.com/imessam/RF-DETR-ONNX/releases/download/models/onnx.zip` | URL to download ONNX models if none are found. |

Results are saved to `benchmarks/results/results.md`. 

### Latest Benchmark Results

**Tested on:**
| Component | Details |
| :--- | :--- |
| **CPU** | AMD Ryzen 5 7600 6-Core Processor |
| **CPU Cores / Threads** | 6 cores / 12 threads |
| **RAM** | 7.3 GB |
| **GPU** | NVIDIA GeForce RTX 4060 (8.0 GB VRAM, driver 591.74) |

#### Model: RF-DETR Nano (384×384)
| Implementation | Device | Preprocess | ORT Run | Postprocess | Total | FPS |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Python | CPU | 2.21 ms | 67.31 ms | 0.37 ms | 69.86 ms | 14.31 |
| Python | GPU | 1.65 ms | 7.83 ms | 0.17 ms | 9.62 ms | 103.93 |
| C++ | CPU | 0.77 ms | 56.15 ms | 0.13 ms | 58.17 ms | 17.19 |
| C++ | GPU | **0.61 ms** | **7.82 ms** | **0.09 ms** | **8.47 ms** | **118.01** 🚀 |

#### Model: RF-DETR Small (512×512)
| Implementation | Device | Preprocess | ORT Run | Postprocess | Total | FPS |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Python | CPU | 2.77 ms | 117.45 ms | 0.29 ms | 120.53 ms | 8.30 |
| Python | GPU | 4.06 ms | 31.79 ms | 0.18 ms | 36.17 ms | 27.65 |
| C++ | CPU | 0.98 ms | 119.17 ms | 0.13 ms | 120.22 ms | 8.32 |
| C++ | GPU | **0.94 ms** | **12.79 ms** | **0.09 ms** | **13.74 ms** | **72.80** 🚀 |

*Note: Best performance values are **highlighted in bold**.*

---

## Documentation

Full documentation is available at: **[imessam.github.io/RF-DETR-ONNX](https://imessam.github.io/RF-DETR-ONNX/)** *(or build locally — see below)*.

```bash
# Install docs dependencies
uv sync --extra docs

# Serve locally
mkdocs serve
# Open http://127.0.0.1:8000
```

---

## License

This repository is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

Parts of this code are derived from Roboflow software:

| License | Applies To |
|---------|-----------|
| [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) | RF-DETR models/weights (except xlarge/2xlarge) and the `rfdetr` Python package |
| [Platform Model License 1.0](https://roboflow.com/platform-model-license-1-0) | `rfdetr-xlarge` and `rfdetr-2xlarge` models and weights |

More information: [roboflow.com/licensing](https://roboflow.com/licensing)

---

## Acknowledgements

- [Roboflow](https://roboflow.com) team for developing RF-DETR and sharing it under a permissive license.
- [PierreMarieCurie](https://github.com/PierreMarieCurie) for the initial ONNX implementation and model conversions.
