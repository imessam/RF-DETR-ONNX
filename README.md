# RF-DETR with ONNX

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-yellow)](https://huggingface.co/PierreMarieCurie/rf-detr-onnx/tree/main)


This repository is a fork of the original work by [PierreMarieCurie](https://github.com/PierreMarieCurie/rf-detr-onnx), reworked and organized into a modular structure with additional features like manual device selection and performance metrics. Special thanks to [PierreMarieCurie](https://github.com/PierreMarieCurie) for the initial implementation and model conversions.

RF-DETR is a transformer-based object detection and instance segmentation architecture developed by Roboflow. For more details on the model, please refer to the impressive work by the Roboflow team [here](https://github.com/roboflow/rf-detr/tree/main).

| Roboflow | ONNX Runtime Inference<p> (Object detection) | ONNX Runtime Inference <p> (Instance segmentation) |
|----------------------|-----------------------------|-----------------------------|
| <p align="center"><img src="assets/official_repo.png" width="100%"></p> | <p align="center"><img src="assets/object_detection.jpg" width="74%"></p> | <p align="center"><img src="assets/instance_segmentation.jpg" width="74%"></p> |

## Project Structure

The project is organized within the `python/` directory:

- `python/inference.py`: High-level script for running inference on images.
- `python/run_validation.sh`: Master script for end-to-end model preparation and validation.
- `python/modules/`: Core logic and modules.
  - `model.py`: High-level detection model class (`RFDETRModel`).
  - `onnx_runtime.py`: ONNX Runtime session management.
  - `utils.py`: Common utility functions.
  - `export.py`: Core export logic.
- `python/tests/`: Quality assurance and validation tools.
  - `prepare_models.py`: Handles weight download and ONNX export.
  - `generate_torch_results.py`: Reference result generator (PyTorch).
  - `generate_onnx_results.py`: Target result generator (ONNX).
  - `test_val.py`: Accuracy comparison test suite.
- `output/`: Default directory for inference results.

## Installation

First, clone the repository:

```bash
git clone https://github.com/imessam/rf-detr-onnx.git
cd rf-detr-onnx/python
```

### Using uv (recommended)

If not installed, run:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For a full development and testing setup (including export and validation tools):
```bash
uv sync --extra export --extra test
```

For lightweight inference only:
```bash
uv sync
```

## Validation & Testing

We provide a fully automated validation pipeline that ensures the exported ONNX model matches the original PyTorch model's accuracy.

### Run Full Pipeline
The master script handles dependency syncing, model preparation, result generation, and accuracy comparison:

```bash
pip install --upgrade .
```
Make sure to install Python 3.10+ on your local or virtual environment.
</details>

## Model to ONNX format

### Downloading from Hugging-face

Roboflow provides pre-trained RF-DETR models on the [COCO](https://cocodataset.org/#home) and [Objects365](https://www.objects365.org/overview.html) datasets. We have already converted some of these models to the ONNX format for you, which you can directly download from [Hugging Face](https://huggingface.co/PierreMarieCurie/rf-detr-onnx).

Note that this corresponds to [rf-detr version 1.4.1](https://github.com/roboflow/rf-detr/tree/1.4.1):
- **Object detection**:
  - Trained on COCO dataset:
    - [rf-detr-nano](https://huggingface.co/PierreMarieCurie/rf-detr-onnx/resolve/main/rf-detr-nano.onnx)
    - [rf-detr-base](https://huggingface.co/PierreMarieCurie/rf-detr-onnx/resolve/main/rf-detr-base-coco.onnx)
    - [rf-detr-small](https://huggingface.co/PierreMarieCurie/rf-detr-onnx/resolve/main/rf-detr-small.onnx)
    - [rf-detr-medium](https://huggingface.co/PierreMarieCurie/rf-detr-onnx/resolve/main/rf-detr-medium.onnx)
    - [rf-detr-large](https://huggingface.co/PierreMarieCurie/rf-detr-onnx/resolve/main/rf-detr-large.onnx) (**DEPRECATED**)
    - [rf-detr-large-2026](https://huggingface.co/PierreMarieCurie/rf-detr-onnx/resolve/main/rf-detr-large-2026.onnx)
    - [rf-detr-xlarge](https://huggingface.co/PierreMarieCurie/rf-detr-onnx/resolve/main/rf-detr-xlarge.onnx)
    - [rf-detr-2xlarge](https://huggingface.co/PierreMarieCurie/rf-detr-onnx/resolve/main/rf-detr-xxlarge.onnx)
  - Trained on Objects365 dataset:
    - [rf-detr-base-o365](https://huggingface.co/PierreMarieCurie/rf-detr-onnx/resolve/main/rf-detr-base-o365.onnx)
- **Instance segmentation** (train on COCO dataset)
    - [rf-detr-seg-preview](https://huggingface.co/PierreMarieCurie/rf-detr-onnx/resolve/main/rf-detr-seg-preview.onnx)
    - [rf-detr-seg-nano](https://huggingface.co/PierreMarieCurie/rf-detr-onnx/resolve/main/rf-detr-seg-nano.onnx)
    - [rf-detr-seg-small](https://huggingface.co/PierreMarieCurie/rf-detr-onnx/resolve/main/rf-detr-seg-small.onnx)
    - [rf-detr-seg-medium](https://huggingface.co/PierreMarieCurie/rf-detr-onnx/resolve/main/rf-detr-seg-medium.onnx)
    - [rf-detr-seg-large](https://huggingface.co/PierreMarieCurie/rf-detr-onnx/resolve/main/rf-detr-seg-large.onnx)
    - [rf-detr-seg-xlarge](https://huggingface.co/PierreMarieCurie/rf-detr-onnx/resolve/main/rf-detr-seg-xlarge.onnx)
    - [rf-detr-seg-2xlarge](https://huggingface.co/PierreMarieCurie/rf-detr-onnx/resolve/main/rf-detr-seg-xxlarge.onnx)

### Converting 

If you want to export your own fine-tuned RF-DETR model to ONNX, you can use the preparation script directly:

```bash
uv sync --extra export
uv run python tests/prepare_models.py
```

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

## Version Compatibility

| Repo Tag | rfdetr Version | Status |
|--------|----------------|--------|
| v1.0-rfdetr1.3.0 | [1.3.0](https://github.com/roboflow/rf-detr/tree/1.3.0) | Stable |
| main | [1.4.1](https://github.com/roboflow/rf-detr/tree/1.4.1) | In progress |

## License

This repository is licensed under the MIT License. See [LICENSE](LICENSE) for more details.

However, some parts of the code are derived from Roboflow software. Below are the details:

- **Apache License 2.0** ([reference](https://www.apache.org/licenses/LICENSE-2.0)): RF-DETR models and pretrained weights (except `rfdetr-xlarge` and `rfdetr-2xlarge`) and all `rfdetr` Python package.
- **Platform Model License 1.0 (PML-1.0)** ([reference](https://roboflow.com/platform-model-license-1-0)): `rfdetr-xlarge` and `rfdetr-2xlarge` models and pretrained weights.

More information about Roboflow model licensing [here](https://roboflow.com/licensing).

## Acknowledgements
- Thanks to the **Roboflow** team and everyone involved in the development of RF-DETR, particularly for sharing a state-of-the-art model under a permissive free software license.