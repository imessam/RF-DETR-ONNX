#!/bin/bash
set -e

# Default Configuration
VERBOSE=0
MODEL_PTH_URL="https://huggingface.co/PierreMarieCurie/rf-detr-onnx/resolve/main/rf-detr-nano.pth.zip"
MODEL_ONNX_URL="https://huggingface.co/PierreMarieCurie/rf-detr-onnx/resolve/main/rf-detr-nano.onnx.zip"
DEVICE="gpu"


usage() {
    echo "Usage: ./run_tests.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -d <string> Device to use (gpu/cpu, default: gpu)"
    echo "  -v          Enable verbose logging"
    echo "  -p <url>    URL for PTH zip (default: $MODEL_PTH_URL)"
    echo "  -n <url>    URL for ONNX zip (default: $MODEL_ONNX_URL)"
    echo "  -o <path>   ONNX Runtime root directory (optional)"


    echo "  -h          Show this help message"

    echo ""
    exit 0
}

# Parse CLI arguments
while getopts "d:p:n:o:vh" opt; do
    case "$opt" in
        d) DEVICE=$OPTARG ;;
        p) MODEL_PTH_URL=$OPTARG ;;
        n) MODEL_ONNX_URL=$OPTARG ;;
        o) ONNX_ROOT=$OPTARG ;;


        v) VERBOSE=1 ;;

        h) usage ;;
        *) usage ;;
    esac
done

# Paths
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_DIR="$REPO_ROOT/tests"
RESULTS_DIR="$TEST_DIR/results"
ASSETS_DIR="$TEST_DIR/assets/images"

VERBOSE_PY_ARG=""
if [ "$VERBOSE" -eq 1 ]; then
    VERBOSE_PY_ARG="--verbose"
fi

CMAKE_ONNX_ARG=""
if [ -n "$ONNX_ROOT" ]; then
    CMAKE_ONNX_ARG="-DONNXRUNTIME_ROOT_DIR=$ONNX_ROOT"
fi


# Clean old results
echo ">>> Cleaning old results..."
rm -rf "$RESULTS_DIR/python_onnx" "$RESULTS_DIR/cpp_onnx" "$RESULTS_DIR/reference"
mkdir -p "$RESULTS_DIR/python_onnx" "$RESULTS_DIR/cpp_onnx" "$RESULTS_DIR/reference"

# Ensure environment is synced
echo ">>> Syncing Python environment..."
cd "$REPO_ROOT"
uv sync --extra test

# 0. Asset Check (Only single image as requested)
if [ ! -d "$ASSETS_DIR" ] || [ -z "$(ls -A "$ASSETS_DIR" 2>/dev/null)" ]; then
    echo ">>> Downloading Single Test Asset..."
    mkdir -p "$ASSETS_DIR"
    curl -o "$ASSETS_DIR/coco_1.jpg" http://images.cocodataset.org/val2017/000000000139.jpg
else
    # Keep only one image if many exist
    FIRST_IMG=$(ls "$ASSETS_DIR" | head -n 1)
    echo ">>> Using single asset: $FIRST_IMG"
    find "$ASSETS_DIR" -type f ! -name "$FIRST_IMG" -delete
fi

# 1. Build C++ Library & Tests
echo ">>> Building C++ Library..."
cd "$REPO_ROOT/cpp"
mkdir -p build && cd build
cmake $CMAKE_ONNX_ARG -DENABLE_TESTS=ON ..
make -j$(nproc)

echo ">>> Building C++ Tests (Standalone Tool)..."
cd "$REPO_ROOT/cpp/tests"
mkdir -p build && cd build
cmake $CMAKE_ONNX_ARG ..
make -j$(nproc)


# 2. Manage Models
echo ">>> Managing Models..."
TORCH_DIR="${TEST_DIR}/models/torch"
ONNX_DIR="${TEST_DIR}/models/onnx"

mkdir -p "$TORCH_DIR" "$ONNX_DIR"

# Check if models exist
TORCH_MODEL=$(ls "$TORCH_DIR"/*.pth 2>/dev/null | head -n 1)
ONNX_MODEL=$(ls "$ONNX_DIR"/*.onnx 2>/dev/null | head -n 1)

# PTH Model Download
if [ -z "$TORCH_MODEL" ]; then
    echo ">>> Downloading PTH model from $MODEL_PTH_URL..."
    TEMP_DIR=$(mktemp -d)
    ZIP_PATH="$TEMP_DIR/pth_models.zip"
    if curl -L -f -o "$ZIP_PATH" "$MODEL_PTH_URL"; then
        echo ">>> Extracting..."
        unzip -q "$ZIP_PATH" -d "$TEMP_DIR"
        find "$TEMP_DIR" -name "*.pth" -exec mv {} "$TORCH_DIR/" \;
        TORCH_MODEL=$(ls "$TORCH_DIR"/*.pth 2>/dev/null | head -n 1)
    else
        echo "⚠️ Warning: Failed to download PTH model from $MODEL_PTH_URL"
    fi
    rm -rf "$TEMP_DIR"
fi

# ONNX Model Download
if [ -z "$ONNX_MODEL" ]; then
    echo ">>> Downloading ONNX model from $MODEL_ONNX_URL..."
    TEMP_DIR=$(mktemp -d)
    ZIP_PATH="$TEMP_DIR/onnx_models.zip"
    if curl -L -f -o "$ZIP_PATH" "$MODEL_ONNX_URL"; then
        echo ">>> Extracting..."
        unzip -q "$ZIP_PATH" -d "$TEMP_DIR"
        find "$TEMP_DIR" -name "*.onnx" -exec mv {} "$ONNX_DIR/" \;
        ONNX_MODEL=$(ls "$ONNX_DIR"/*.onnx 2>/dev/null | head -n 1)
    else
        echo "⚠️ Warning: Failed to download ONNX model from $MODEL_ONNX_URL"
    fi
    rm -rf "$TEMP_DIR"
fi



if [ -z "$TORCH_MODEL" ] || [ -z "$ONNX_MODEL" ]; then
    echo "❌ Error: Required models (.pth and .onnx) not found."
    exit 1
fi

echo ">>> Using Torch model: $TORCH_MODEL"
echo ">>> Using ONNX model: $ONNX_MODEL"

# 3. Generate Reference (Torch)
echo ">>> Generating Torch Reference..."
cd "$REPO_ROOT"
uv run python tests/generate_reference.py --weights "$TORCH_MODEL" --assets_dir "$ASSETS_DIR" --output_dir "$RESULTS_DIR/reference" --device "$DEVICE"

# 4. Generate Python ONNX Results
echo ">>> Generating Python ONNX results..."
uv run python python/tests/generate_onnx.py --model "$ONNX_MODEL" --input "$ASSETS_DIR" --device "$DEVICE" --output "$RESULTS_DIR/python_onnx" $VERBOSE_PY_ARG

# 5. Generate C++ ONNX Results
echo ">>> Generating C++ ONNX results..."
"$REPO_ROOT/cpp/tests/build/generate_onnx" "$ONNX_MODEL" "$ASSETS_DIR" "$DEVICE" "$RESULTS_DIR/cpp_onnx"

# 6. Run Accuracy Verification
echo ">>> Running Accuracy Verification..."
uv run pytest tests/test_equivalence.py --reference "$RESULTS_DIR/reference" --python_onnx "$RESULTS_DIR/python_onnx" --cpp_onnx "$RESULTS_DIR/cpp_onnx"

# 7. Generate Final Report
echo ">>> Generating Summary Report..."
uv run python tests/generate_report.py --results-dir "$RESULTS_DIR" --output "$RESULTS_DIR/accuracy_report.md"

echo ""
echo ">>> All Tests Complete!"
echo "Report saved to: $RESULTS_DIR/accuracy_report.md"
