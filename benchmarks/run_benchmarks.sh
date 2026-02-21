#!/bin/bash
set -e

# Default Configuration
ITERATIONS=10
COOLDOWN_SECONDS=2
SLEEP_PER_IMAGE=0.1
VERBOSE=0
MODEL_ZIP_URL="https://github.com/imessam/RF-DETR-ONNX/releases/download/models/onnx.zip"

usage() {
    echo "Usage: ./run_benchmarks.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -n <int>    Number of iterations (default: 10)"
    echo "  -c <float>  Cooldown seconds between benchmarks (default: 2)"
    echo "  -s <float>  Sleep per image iteration (default: 0.1)"
    echo "  -v          Enable verbose per-iteration logging"
    echo "  -u <url>    URL to download ONNX zip if none found (default: $MODEL_ZIP_URL)"
    echo "  -h          Show this help message"
    echo ""
    echo "Example:"
    echo "  ./run_benchmarks.sh -n 5 -v -c 1"
    exit 0
}

# Parse CLI arguments
while getopts "n:c:s:u:vh" opt; do
    case "$opt" in
        n) ITERATIONS=$OPTARG ;;
        c) COOLDOWN_SECONDS=$OPTARG ;;
        s) SLEEP_PER_IMAGE=$OPTARG ;;
        u) MODEL_ZIP_URL=$OPTARG ;;
        v) VERBOSE=1 ;;
        h) usage ;;
        *) usage ;;
    esac
done

MODELS_DIR="models"
IMAGES_DIR="benchmarks/assets/images"

# Paths
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_DIR="$REPO_ROOT/benchmarks"
RESULTS_DIR="$BENCH_DIR/results"

VERBOSE_PY_ARG=""
VERBOSE_CPP_ARG=""
if [ "$VERBOSE" -eq 1 ]; then
    VERBOSE_PY_ARG="--verbose"
    VERBOSE_CPP_ARG="verbose"
fi

# Clean old results
echo ">>> Cleaning old results..."
rm -rf "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR"

# Ensure environment is synced with GPU support
echo ">>> Syncing Python environment..."
cd "$REPO_ROOT"
uv sync

echo ">>> Starting Benchmarks with $ITERATIONS iterations..."

# 0. Ensure ONNX Models Exist (prefer root models/, else download zip)
if [ ! -d "$REPO_ROOT/$MODELS_DIR" ] || [ -z "$(find "$REPO_ROOT/$MODELS_DIR" -type f -name '*.onnx' 2>/dev/null)" ]; then
    echo "0 >>> No ONNX models found in $MODELS_DIR. Downloading from $MODEL_ZIP_URL..."
    TEMP_DIR=$(mktemp -d)
    ZIP_PATH="$TEMP_DIR/onnx_models.zip"
    if curl -L -f -o "$ZIP_PATH" "$MODEL_ZIP_URL"; then
        echo "0 >>> Extracting ONNX models..."
        mkdir -p "$REPO_ROOT/$MODELS_DIR"
        unzip -q "$ZIP_PATH" -d "$REPO_ROOT/$MODELS_DIR"
    else
        echo "❌ Error: Failed to download ONNX models from $MODEL_ZIP_URL"
        rm -rf "$TEMP_DIR"
        exit 1
    fi
    rm -rf "$TEMP_DIR"
else
    echo "✓ ONNX models already exist in $MODELS_DIR"
fi

# 1. Asset Check
if [ ! -d "$BENCH_DIR/assets/images" ]; then
    echo "1 >>> Downloading Benchmarking Assets..."
    mkdir -p "$BENCH_DIR/assets/images"
    curl -o "$BENCH_DIR/assets/images/coco_1.jpg" http://images.cocodataset.org/val2017/000000000139.jpg
    curl -o "$BENCH_DIR/assets/images/coco_2.jpg" http://images.cocodataset.org/val2017/000000000285.jpg
    curl -o "$BENCH_DIR/assets/images/coco_3.jpg" http://images.cocodataset.org/val2017/000000000632.jpg
fi

# 2. Build C++ Library & Benchmark
echo "2 >>> Building C++ Library..."
cd "$REPO_ROOT/cpp"
mkdir -p build
cd build
cmake ..
make -j$(nproc)

echo ">>> Building C++ Benchmark..."
cd "$REPO_ROOT/cpp/benchmarks"
mkdir -p build
cd build
cmake ..
make -j$(nproc)

# 3. Discover Models
echo "3 >>> Discovering ONNX models..."
cd "$REPO_ROOT"

# Find all .onnx files in root models directory (recursive)
if [ ! -d "$MODELS_DIR" ] || [ -z "$(find "$MODELS_DIR" -type f -name '*.onnx' 2>/dev/null)" ]; then
    echo "❌ Error: No ONNX models found in $MODELS_DIR"
    echo "Please add .onnx model files to $MODELS_DIR/ or provide a valid ONNX zip URL"
    exit 1
fi

# Get list of models (stable order)
mapfile -t MODELS < <(find "$MODELS_DIR" -type f -name '*.onnx' | sort)
echo "Found ${#MODELS[@]} model(s):"
for model in "${MODELS[@]}"; do
    echo "  - $(basename "$model")"
done

# 4. Run Benchmarks for Each Model
test_inputs=("images:$IMAGES_DIR")

for model_path in "${MODELS[@]}"; do
    model_name=$(basename "$model_path" .onnx)
    echo ""
    echo "========================================="
    echo "Benchmarking Model: $model_name"
    echo "========================================="
    
    for test in "${test_inputs[@]}"; do
        type="${test%%:*}"
        path="${test#*:}"
        
        # Python CPU
        echo ">>> Resting for $COOLDOWN_SECONDS seconds..."
        sleep "$COOLDOWN_SECONDS"
        echo "4 >>> Running Python $type Benchmark (CPU)..."
        uv run python/benchmarks/benchmark.py --model "$model_path" --input "$REPO_ROOT/$path" --device cpu --iterations "$ITERATIONS" --sleep-per-image "$SLEEP_PER_IMAGE" $VERBOSE_PY_ARG --output "$RESULTS_DIR/python_cpu_${type}_${model_name}.json"
        
        # Python GPU
        echo ">>> Resting for $COOLDOWN_SECONDS seconds..."
        sleep "$COOLDOWN_SECONDS"
        echo "4 >>> Running Python $type Benchmark (GPU)..."
        uv run python/benchmarks/benchmark.py --model "$model_path" --input "$REPO_ROOT/$path" --device gpu --iterations "$ITERATIONS" --sleep-per-image "$SLEEP_PER_IMAGE" $VERBOSE_PY_ARG --output "$RESULTS_DIR/python_gpu_${type}_${model_name}.json"
        
        # C++ CPU
        echo ">>> Resting for $COOLDOWN_SECONDS seconds..."
        sleep "$COOLDOWN_SECONDS"
        echo "4 >>> Running C++ $type Benchmark (CPU)..."
        "$REPO_ROOT/cpp/benchmarks/build/rfdetr_benchmark" "$model_path" "$REPO_ROOT/$path" cpu "$ITERATIONS" "$SLEEP_PER_IMAGE" $VERBOSE_CPP_ARG
        mv benchmark_cpp_cpu.json "$RESULTS_DIR/cpp_cpu_${type}_${model_name}.json"
        
        # C++ GPU
        echo ">>> Resting for $COOLDOWN_SECONDS seconds..."
        sleep "$COOLDOWN_SECONDS"
        echo "4 >>> Running C++ $type Benchmark (GPU)..."
        "$REPO_ROOT/cpp/benchmarks/build/rfdetr_benchmark" "$model_path" "$REPO_ROOT/$path" gpu "$ITERATIONS" "$SLEEP_PER_IMAGE" $VERBOSE_CPP_ARG
        mv benchmark_cpp_gpu.json "$RESULTS_DIR/cpp_gpu_${type}_${model_name}.json"
    done
done

# 5. Generate Report
echo "5 >>> Generating Results Report..."
cd "$BENCH_DIR"
uv run python generate_report.py --results-dir results --output results/results.md

echo ""
echo ">>> All Benchmarks Complete!"
echo "Results saved to: $BENCH_DIR/results/"
echo "Report saved to: $BENCH_DIR/results/results.md"
