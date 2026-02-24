# Tools

The `tools/` directory contains scripts for exporting RF-DETR models from PyTorch checkpoints to ONNX format.

---

## `export_roboflow.py`

Export a Roboflow fine-tuned RF-DETR checkpoint (`.pth`) to ONNX.

### Requirements

Install the `export` extra:

```bash
uv sync --extra export
# or
pip install ".[export]"
```

### Usage

```bash
uv run python tools/export_roboflow.py \
    --weights path/to/checkpoint.pth \
    --model-type nano \
    --output-dir models/ \
    --opset 17
```

### Parameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--weights` | `str` | **Required** | Path to the `.pth` checkpoint file |
| `--model-type` | `str` | `nano` | Architecture: `nano`, `small`, `base`, `medium`, `large` |
| `--output-dir` | `str` | `models/` | Directory to save the exported ONNX model |
| `--opset` | `int` | `17` | ONNX opset version |
| `--no-simplify` | flag | off | Disable `onnxsim` simplification step |

### Output

The script produces a simplified ONNX model under a folder named after the checkpoint file:

```
<output-dir>/<checkpoint-stem>/<checkpoint-stem>.onnx
```

For example, if `--weights rf-detr-nano.pth` and `--output-dir models/`, the output is:

```
models/rf-detr-nano/rf-detr-nano.onnx
```

!!! tip "Model Simplification"
    By default, `onnxsim` is applied to the exported graph. This folds constants and removes redundant nodes, resulting in faster and more portable models.

---

## `export.py`

Low-level ONNX export utility. Auto-detects the model architecture from checkpoint weights and exports to ONNX. Supports all model variants including segmentation and XLarge models.

### Usage

```bash
uv run python tools/export.py \
    --checkpoint path/to/checkpoint.pth \
    --model-name output.onnx
```

### Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | **Required** | Path to the `.pth` or `.pt` checkpoint file |
| `--model-name` | *(derived from checkpoint name)* | Output ONNX filename |
| `--no-simplify` | off | Disable `onnxsim` simplification |

!!! note
    `export.py` auto-detects the model class by counting backbone parameters. Use `export_roboflow.py` instead when you want to explicitly specify the architecture type.

---

## `download_onnx.sh`

Downloads the **ONNX Runtime C++ library** (not models) for Linux x64. Useful when you need to manually install the ONNX Runtime C++ library for the C++ build.

```bash
# Download CPU version (default)
bash tools/download_onnx.sh

# Download GPU version
bash tools/download_onnx.sh -d gpu

# Specify a version and output directory
bash tools/download_onnx.sh -v 1.21.0 -d gpu -o libs/onnx
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `-v <version>` | `1.21.0` | ONNX Runtime version to download |
| `-d <device>` | `cpu` | Device type: `cpu` or `gpu` |
| `-o <dir>` | `libs/onnx` | Output directory |

After downloading, pass the extracted directory to CMake:

```bash
cmake .. -DONNXRUNTIME_ROOT_DIR=$(pwd)/libs/onnx/onnxruntime-linux-x64-1.21.0
```

!!! tip "Pre-converted ONNX models"
    To download pre-converted RF-DETR ONNX **models** (not the runtime), run the benchmark or test scripts — they auto-download models from the GitHub release. Or download manually from [Hugging Face](https://huggingface.co/PierreMarieCurie/rf-detr-onnx/tree/main).

---

## Supported Model Types

| Type | Parameters | Input Size | Notes |
|------|-----------|-----------|-------|
| `nano` | ~6M | 384×384 | Fastest, recommended for edge/CPU |
| `small` | ~12M | 384×384 | Good speed/accuracy balance |
| `base` | ~32M | 560×560 | Strong accuracy |
| `medium` | ~50M | 560×560 | High accuracy |
| `large` | ~100M+ | 560×560 | Highest accuracy, GPU recommended |
