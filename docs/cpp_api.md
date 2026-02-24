# C++ API Reference

The C++ library lives in the `cpp/` directory and exposes a `rfdetr` namespace. It is designed to be consumed as a static library via CMake.

---

## Namespace: `rfdetr`

---

## Structs

### `Detection`

Holds a single detected object.

```cpp
struct Detection {
    float     score;              // Confidence score [0, 1]
    int       label;              // Predicted class index
    cv::Rect2f normalizedBox;     // [x, y, w, h] normalized to [0, 1]
    cv::Rect2f unnormalizedBox;   // [x, y, w, h] in pixel coordinates
    cv::Mat    mask;              // Binary segmentation mask (empty if detection-only)
};
```

---

### `Timings`

Holds per-stage inference timing in milliseconds.

```cpp
struct Timings {
    float preprocess   = 0.0f;  // Image resize + normalize
    float ortRun       = 0.0f;  // ONNX Runtime execution
    float postprocess  = 0.0f;  // Score filter + box decode + mask resize
    float total        = 0.0f;  // End-to-end wall time
};
```

---

## Class: `RFDETRModel`

High-level inference interface. Handles the complete pipeline: preprocessing → ONNX Runtime → postprocessing.

**Header:** [`cpp/include/rfdetr_model.hpp`](https://github.com/imessam/RF-DETR-ONNX/blob/main/cpp/include/rfdetr_model.hpp)

### Constructor

```cpp
RFDETRModel(const std::string& modelPath,
            const std::string& device = "gpu");
```

| Parameter | Description |
|-----------|-------------|
| `modelPath` | Path to the `.onnx` model file |
| `device` | Device: `"gpu"` (tries TensorRT → CUDA → CPU in order) or `"cpu"` |

---

### `predict`

```cpp
void predict(const cv::Mat&           image,
             std::vector<Detection>&  detections,
             Timings&                 timings,
             float confidenceThreshold = 0.5f,
             int   maxNumberBoxes      = 300);
```

Runs end-to-end inference.

| Parameter | Description |
|-----------|-------------|
| `image` | Input image in BGR format (OpenCV `cv::Mat`) |
| `detections` | Output: populated with detected objects |
| `timings` | Output: populated with per-stage timing |
| `confidenceThreshold` | Minimum confidence to keep a detection |
| `maxNumberBoxes` | Maximum number of detections returned |

---

### `saveDetections`

```cpp
void saveDetections(const cv::Mat&                  image,
                    const std::vector<Detection>&   detections,
                    const std::string&              savePath) const;
```

Draws bounding boxes, masks, and labels on the image and saves it.

| Parameter | Description |
|-----------|-------------|
| `image` | Original BGR image |
| `detections` | Detection results from `predict()` |
| `savePath` | Output file path (e.g., `"output/result.jpg"`) |

---

### `warmup`

```cpp
void warmup();
```

Performs a dummy inference pass to initialize GPU/TensorRT resources. Called automatically during construction when a GPU provider is active.

---

## Class: `OnnxRuntimeSession`

Low-level ONNX Runtime session wrapper.

**Header:** [`cpp/include/onnx_runtime.hpp`](https://github.com/imessam/RF-DETR-ONNX/blob/main/cpp/include/onnx_runtime.hpp)

### Constructor

```cpp
OnnxRuntimeSession(const std::string& modelPath,
                   const std::string& device = "gpu");
```

### Key Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `run(input)` | `std::vector<cv::Mat>` | Run inference; returns output tensors as `cv::Mat` |
| `getInputShape()` | `std::vector<int64_t>` | Expected input shape `[N, C, H, W]` |
| `getInputName()` | `std::string` | Name of the input tensor |

## Usage Example

```cpp
#include "rfdetr_model.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    rfdetr::RFDETRModel model("models/rf-detr-nano/rf-detr-nano.sim.onnx", "gpu");

    cv::Mat image = cv::imread("assets/drone.jpg");

    std::vector<rfdetr::Detection> detections;
    rfdetr::Timings timings;

    model.predict(image, detections, timings, 0.5f, 300);

    std::cout << "Found " << detections.size() << " detections\n";
    std::cout << "ORT run: " << timings.ortRun << " ms\n";

    model.saveDetections(image, detections, "output/result.jpg");
    return 0;
}
```

---

## CMake Integration

```cmake
find_package(rfdetr_onnx REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE rfdetr_onnx::rfdetr_onnx)
```

Install the library first:

```bash
cd cpp && mkdir build && cd build
cmake .. -DONNXRUNTIME_ROOT_DIR=/path/to/onnxruntime
make -j$(nproc)
cmake --install .
```
