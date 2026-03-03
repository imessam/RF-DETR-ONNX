# RF-DETR ONNX

<p align="center">
  <img src="assets/detection_demo.jpg" alt="RF-DETR detection demo" width="600">
</p>

**RF-DETR ONNX** is a modular, production-ready library for running [RF-DETR](https://github.com/roboflow/rf-detr) object detection and instance segmentation with [ONNX Runtime](https://onnxruntime.ai/) — in both Python and C++.

Developed as a fork of [PierreMarieCurie/rf-detr-onnx](https://github.com/PierreMarieCurie/rf-detr-onnx), it adds a modular architecture, manual device selection, performance metrics, and a high-performance C++ library.

---

## Features

- ✅ **Python & C++ implementations** — from prototyping to production
- ⚡ **Automatic provider selection** — TensorRT → CUDA → CPU fallback
- 📊 **Detailed timing breakdown** — preprocess, inference, postprocess
- 🔧 **Optimized C++ implementation** — high-performance library for production deployment
- 🎯 **Instance segmentation** — supports models with mask outputs
- 🏎️ **FP16 & mixed-precision support** — convert models to float16 for ~2× GPU speedup and half the memory footprint
- 📦 **Modular library** — install as a static C++ library or Python package

---

## Quick Start

```bash
git clone https://github.com/imessam/rf-detr-onnx.git
cd rf-detr-onnx
uv sync
uv run python python/inference.py \
    --model models/rf-detr-nano/rf-detr-nano.sim.onnx \
    --image assets/drone.jpg \
    --device cpu
```

See [Getting Started → Quick Start](quickstart.md) for more usage examples.

---

## Demo Results

| Original Image | Torch Reference | ONNX Inference |
|----------------|-----------------|----------------|
| ![Original](assets/drone.jpg){ width=100% } | ![Reference](assets/reference_demo.jpg){ width=100% } | ![ONNX Result](assets/detection_demo.jpg){ width=100% } |

---

## Project Layout

```
rf-detr-onnx/
├── python/             # Python library and CLI
│   ├── inference.py    # CLI demo script
│   └── modules/        # RFDETRModel, OnnxRuntimeSession, utils
├── cpp/                # C++ library
│   ├── include/        # Header files
│   ├── src/            # Source files
│   └── CMakeLists.txt  # Build system
├── benchmarks/         # Automated benchmarking suite
├── tools/              # Export and model conversion scripts
├── tests/              # Accuracy validation pipeline
└── assets/             # Demo images
```

---

## Links

- [Installation](installation.md)
- [Quick Start](quickstart.md)
- [Python API Reference](python_api.md)
- [C++ API Reference](cpp_api.md)
- [Benchmarks](benchmarks.md)
- [Source on GitHub](https://github.com/imessam/RF-DETR-ONNX)
