# RF-DETR with ONNX

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-yellow)](https://huggingface.co/PierreMarieCurie/rf-detr-onnx/tree/main)


This repository is a fork of the original work by [PierreMarieCurie](https://github.com/PierreMarieCurie/rf-detr-onnx), reworked and organized into a modular structure with additional features like manual device selection and performance metrics. Special thanks to [PierreMarieCurie](https://github.com/PierreMarieCurie) for the initial implementation and model conversions.

RF-DETR is a transformer-based object detection and instance segmentation architecture developed by Roboflow. For more details on the model, please refer to the impressive work by the Roboflow team [here](https://github.com/roboflow/rf-detr/tree/main).

| Original Image | Torch Reference | ONNX Inference Result |
|----------------|-----------------|-----------------------|
| <p align="center"><img src="assets/drone.jpg" width="100%"></p> | <p align="center"><img src="assets/reference_demo.jpg" width="100%"></p> | <p align="center"><img src="assets/detection_demo.jpg" width="100%"></p> |

## Project Structure

The project is organized within the `python/` directory:

- `python/inference.py`: High-level inference demo script.
- `benchmarks/`: Automated performance measurement suite.
  - `run_benchmarks.sh`: Master benchmark script (runs Python & C++ tests).
  - `generate_report.py`: Aggregates JSON results into a Markdown report.
  - `results/`: Output directory for benchmark artifacts.
- `python/modules/`: Core logic and modules.
  - `model.py`: High-level detection model class (`RFDETRModel`).
  - `onnx_runtime.py`: ONNX Runtime session management.
  - `utils.py`: Common utility functions.
  - `export_roboflow.py`: Script to convert RF-DETR checkpoints to ONNX via Roboflow API.
- `python/tests/`: Quality assurance and validation tools.
  - `prepare_models.py`: Handles weight download and ONNX export.
  - `generate_torch_results.py`: Reference result generator (PyTorch).
  - `generate_onnx_results.py`: Target result generator (ONNX).
  - `test_val.py`: Accuracy comparison test suite.
- `output/`: Default directory for inference results.
- `cpp/`: Modular C++ implementation of RF-DETR.
  - `include/`: Header files for model, session, and utils.
  - `src/`: Source code implementation.
  - `CMakeLists.txt`: Build configuration.

## Installation

First, clone the repository:

```bash
git clone https://github.com/imessam/rf-detr-onnx.git
cd rf-detr-onnx/python
```

### Using uv

First, install [uv](https://docs.astral.sh/uv/) if you haven't already:

- **Lightweight Inference (CPU)**:
  ```bash
  uv sync
  ```
- **GPU Acceleration**:
  ```bash
  uv sync --extra gpu
  ```
- **Full Development (Export & Testing)**:
  ```bash
  uv sync --extra export --extra test
  ```

## Validation & Testing

We provide a fully automated validation pipeline that ensures the exported ONNX model matches the original PyTorch model's accuracy.

### Run Full Pipeline
The master script handles dependency syncing, model preparation, result generation, and accuracy comparison:

```bash
./run_validation.sh nano
```

## Converting 

To export your own fine-tuned RF-DETR model to ONNX, use the `export_roboflow.py` script. You'll need the `[export]` extra:

```bash
uv sync --extra export
```bash
uv sync --extra export
uv run python modules/export_roboflow.py --model-type nano
```

#### Export Parameters
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--weights` | `str` | **Required** | Path to the `.pth` checkpoint file. |
| `--model-type` | `str` | `nano` | Architecture type (`nano`, `small`, `base`, `medium`, `large`). |
| `--output-dir` | `str` | `models/` | Directory for the exported model. |
| `--opset` | `int` | `17` | ONNX opset version. |
| `--no-simplify`| `flag`| `False` | Disable model simplification. |

## Inference

### Inference Script

```bash
# Run on CPU (default)
uv run python inference.py --model tests/test_models/inference_model.sim.onnx --image ../assets/drone.jpg

# Run on GPU
uv run python inference.py --model tests/test_models/inference_model.sim.onnx --image ../assets/drone.jpg --device gpu
```

### Programmatic Usage

```python
from modules.model import RFDETRModel

# Initialize the model
model = RFDETRModel("path/to/model.onnx", device="cpu")

# Run inference
scores, labels, boxes, masks = model.predict("path/to/image.jpg")

# Visualize results
model.save_detections("path/to/image.jpg", boxes, labels, masks, "output/result.jpg")
```

## C++ Implementation

The C++ implementation provides a high-performance, modular library for RF-DETR inference.

### Build Instructions

Requires: OpenCV 4.x and ONNX Runtime C++ API.

```bash
cd cpp
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Optimization: Zero-Copy Inference

The C++ implementation is optimized for high-throughput inference by using **zero-copy output handling**. Instead of copying inference results from ONNX Runtime memory, the `OnnxRuntimeSession` wraps the raw output tensors directly into `cv::Mat` objects. This significantly reduces CPU overhead, especially when working with high-resolution segmentation masks.

### Usage

```bash
./rfdetr_onnx_demo \
    --model ../models/rf-detr-nano/rf-detr-nano.sim.onnx \
    --image ../assets/drone.jpg \
    --device gpu
```

## Performance Comparison

Inference performance measured on **RF-DETR Nano** (384x384) using a laptop with **CUDA Acceleration** (RTX 40-series) and a modern multi-core CPU.

## Benchmarking

Performance results and automated benchmarking tools are available in the [benchmarks/](benchmarks/) directory. The suite evaluates both Python and C++ implementations across CPU and GPU providers.

### Running Benchmarks

1. **Prepare Models**: Ensure your `.onnx` models are in `models/onnx/`.
2. **Execute Suite**:
   ```bash
   cd benchmarks
   ./run_benchmarks.sh -n 10 -v
   ```

#### Options
| Option | Default | Description |
|--------|---------|-------------|
| `-n <int>` | `10` | Number of benchmark iterations. |
| `-c <float>`| `2.0`| Cooldown period (seconds) between runs. |
| `-v` | `Off` | Enable verbose per-iteration logging. |
| `-u <url>` | `None`| URL to download a model if none are found. |

The script will build the C++ components, run inference across all discovered models, and generate a detailed report in `benchmarks/results/results.md`.


## License

This repository is licensed under the MIT License. See [LICENSE](LICENSE) for more details.

However, some parts of the code are derived from Roboflow software. Below are the details:

- **Apache License 2.0** ([reference](https://www.apache.org/licenses/LICENSE-2.0)): RF-DETR models and pretrained weights (except `rfdetr-xlarge` and `rfdetr-2xlarge`) and all `rfdetr` Python package.
- **Platform Model License 1.0 (PML-1.0)** ([reference](https://roboflow.com/platform-model-license-1-0)): `rfdetr-xlarge` and `rfdetr-2xlarge` models and pretrained weights.

More information about Roboflow model licensing [here](https://roboflow.com/licensing).

## Acknowledgements
- Thanks to the **Roboflow** team and everyone involved in the development of RF-DETR, particularly for sharing a state-of-the-art model under a permissive free software license.