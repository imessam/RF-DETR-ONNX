# Installation

## Requirements

- Python 3.10+
- (Optional) CUDA-capable GPU for accelerated inference

---

## Using uv (Recommended)

[uv](https://docs.astral.sh/uv/) is the recommended package manager. Install it first, then:

```bash
git clone https://github.com/imessam/rf-detr-onnx.git
cd rf-detr-onnx
```

`onnxruntime-gpu` is a **base dependency** — GPU acceleration is enabled by default with `uv sync`. There is no separate `--extra gpu`.

```bash
# Default install (GPU via onnxruntime-gpu)
uv sync

# With model export support
uv sync --extra export

# With test support
uv sync --extra test

# With documentation tools
uv sync --extra docs
```

!!! tip "CPU-only override"
    If you don't have a GPU, override `onnxruntime-gpu` after syncing:

    ```bash
    uv sync
    .venv/bin/pip uninstall -y onnxruntime-gpu
    .venv/bin/pip install onnxruntime
    ```

---

## Using pip

From the repository root:

```bash
pip install .
```

With optional extras:

```bash
pip install ".[export]"   # Model export from .pth checkpoints
pip install ".[test]"     # Accuracy testing with pytest
pip install ".[docs]"     # Build documentation locally
```

!!! tip "CPU-only installation"
    If you don't have a GPU and `onnxruntime-gpu` fails to install, replace it with the CPU version:

    ```bash
    pip uninstall -y onnxruntime-gpu
    pip install onnxruntime
    ```

---

## C++ Build Requirements

The C++ implementation requires:

| Dependency | Version | Notes |
|------------|---------|-------|
| CMake | ≥ 3.15 | Build system |
| OpenCV | ≥ 4.x | Image processing |
| ONNX Runtime C++ | ≥ 1.17 | Inference engine |

### Build

```bash
cd cpp
mkdir build && cd build
cmake ..
make -j$(nproc)
```

!!! note
    For GPU support, ensure the ONNX Runtime C++ library you link against was built with CUDA/TensorRT support.

---

## Downloading Models

Pre-converted ONNX models are available from:

- **[Hugging Face](https://huggingface.co/PierreMarieCurie/rf-detr-onnx/tree/main)**: Download manually.
- **Automatic download**: The benchmark script will download models automatically if none are found.

```bash
# Manual download helper
bash tools/download_onnx.sh
```

Place model files in the `models/` directory (any subfolder is OK).
