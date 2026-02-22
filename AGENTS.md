# Agent Guide: RF-DETR-ONNX

## Project Overview
This repository provides ONNX-based inference for RF-DETR (Roboflow's transformer-based object detection and instance segmentation model). It includes:
- A modular Python implementation for prototyping and validation.
- A high-performance C++ implementation with zero-copy optimizations.
- Benchmarking and tooling for export, conversion, and performance analysis.

Key goals:
- Reliable ONNX export and parity with PyTorch accuracy.
- Fast inference on CPU and GPU.
- Clean separation between Python, C++, benchmarks, and tooling.

## Repository Layout
- `python/`: Python implementation, modules, tests, and validation scripts.
- `cpp/`: C++ implementation (headers, sources, CMake build).
- `benchmarks/`: Benchmark suite and report generation.
- `tools/`: Model export and conversion utilities.
- `models/`: Model artifacts (checkpoints / ONNX outputs).
- `assets/`: Example images.
- `tests/`: Top-level test assets.

## How To Run Common Tasks
- Python inference:
  - `cd python`
  - `uv sync` (or `uv sync --extra gpu`)
  - `uv run python inference.py --model <model.onnx> --image <image>`
- Validation pipeline:
  - `cd python`
  - `./run_validation.sh nano`
- Export from Roboflow checkpoints:
  - `uv sync --extra export`
  - `uv run python tools/export_roboflow.py --model-type nano --weights <path>`
- C++ build:
  - `cd cpp`
  - `mkdir -p build && cd build`
  - `cmake .. && make -j$(nproc)`
- Benchmarks:
  - `cd benchmarks`
  - `./run_benchmarks.sh -n 10 -v`

## Conventions And Expectations
- Keep Python code modular; place reusable logic in `python/modules/`.
- Prefer ONNX Runtime APIs for inference and keep device selection explicit.
- Avoid changing export defaults unless necessary; update docs if you do.
- Add tests or validation steps for changes that affect model outputs.
- Preserve zero-copy handling in C++ unless you have a measured reason to change it.
- For benchmarking and testing: use models in the root `models/` folder if they exist.
- If models are missing, download:
  - PyTorch checkpoints: `https://github.com/imessam/RF-DETR-ONNX/releases/download/models/torch.zip`
  - ONNX models: `https://github.com/imessam/RF-DETR-ONNX/releases/download/models/onnx.zip`

## Where To Look First
- Python inference flow: `python/inference.py`, `python/modules/`
- C++ entry point: `cpp/src/`
- Export pipeline: `tools/export_roboflow.py`, `tools/export.py`
- Validation: `python/run_validation.sh`, `python/tests/`

## If You Change Behavior
- Update `README.md` to reflect new flags, outputs, or workflows.
- Provide a quick validation or benchmark note in your PR/summary.
