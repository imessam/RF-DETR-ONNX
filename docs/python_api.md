# Python API Reference

Auto-generated API documentation for all Python modules in the `python/modules/` package.

---

## `modules.model`

The main high-level module for running RF-DETR inference.

::: modules.model.Detection
    options:
      show_root_heading: true

::: modules.model.RFDETRModel
    options:
      show_root_heading: true
      members:
        - __init__
        - predict
        - save_detections

---

## `modules.onnx_runtime`

Low-level ONNX Runtime session wrapper.

::: modules.onnx_runtime.OnnxRuntimeSession
    options:
      show_root_heading: true
      members:
        - __init__
        - run
        - get_input_shape
        - get_input_name
        - get_inputs

---

## `modules.utils`

Utility functions for image loading and coordinate conversions.

::: modules.utils
    options:
      show_root_heading: true
      members:
        - open_image
        - sigmoid
        - box_cxcywh_to_xywh
        - box_cxcywh_to_xyxyn

---

## Data Structures

### `Detection`

A dataclass representing a single detection result:

| Field | Type | Description |
|-------|------|-------------|
| `score` | `float` | Confidence score in `[0, 1]` |
| `label` | `int` | Predicted class index |
| `normalized_box` | `np.ndarray` | Bounding box `[x, y, w, h]` normalized to `[0, 1]` |
| `unnormalized_box` | `np.ndarray` | Bounding box `[x, y, w, h]` in pixels |
| `mask` | `np.ndarray \| None` | Binary segmentation mask (H, W), or `None` for detection-only models |

### Timings Dictionary

`predict()` returns a `dict[str, float]` with timing in **milliseconds**:

| Key | Description |
|-----|-------------|
| `"preprocess"` | Image resize + normalize + tensor creation |
| `"ort_run"` | ONNX Runtime session execution |
| `"postprocess"` | Score filtering, box decoding, mask resize |
| `"total"` | End-to-end wall time including I/O |

---

## Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `DEFAULT_CONFIDENCE_THRESHOLD` | `0.5` | Default score cutoff |
| `DEFAULT_MAX_NUMBER_BOXES` | `300` | Max detections returned |
