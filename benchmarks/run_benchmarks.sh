#!/bin/bash
set -e

# Configuration
ITERATIONS=${1:-10}
MODEL_URL="${MODEL_URL:-}"  # Set this environment variable or edit below
MODELS_DIR="models/onnx"
IMAGES_DIR="benchmarks/assets/images"

# Paths
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_DIR="$REPO_ROOT/benchmarks"
RESULTS_DIR="$BENCH_DIR/results"

# Clean old results
echo ">>> Cleaning old results..."
rm -rf "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR"

# Ensure environment is synced with GPU support
echo ">>> Syncing Python environment..."
cd "$REPO_ROOT"
uv sync

echo ">>> Starting Benchmarks with $ITERATIONS iterations..."

# 0. Download Model (if MODELS_DIR is empty and MODEL_URL is set)
# Check if there are any .onnx files in models/onnx
if [ ! -d "$REPO_ROOT/$MODELS_DIR" ] || [ -z "$(ls -A "$REPO_ROOT/$MODELS_DIR"/*.onnx 2>/dev/null)" ]; then
    if [ -n "$MODEL_URL" ]; then
        # Extract filename from URL or use a default
        MODEL_NAME_FROM_URL="${MODEL_URL##*/}"
        if [[ ! "$MODEL_NAME_FROM_URL" == *.onnx ]]; then
            MODEL_NAME_FROM_URL="model.onnx"
        fi
        DOWNLOAD_PATH="$MODELS_DIR/$MODEL_NAME_FROM_URL"
        
        echo "0 >>> Downloading model from $MODEL_URL to $DOWNLOAD_PATH..."
        mkdir -p "$(dirname "$REPO_ROOT/$DOWNLOAD_PATH")"
        curl -L -o "$REPO_ROOT/$DOWNLOAD_PATH" "$MODEL_URL"
        
        if [ $? -ne 0 ]; then
            echo "❌ Error: Failed to download model from $MODEL_URL"
            exit 1
        fi
        echo "✓ Model downloaded successfully"
    else
        echo "⚠️ Warning: No models found in $MODELS_DIR and MODEL_URL is not set"
    fi
else
    echo "✓ Models already exist in $MODELS_DIR"
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
MODELS_DIR="models/onnx"

# Find all .onnx files in models/onnx directory
if [ ! -d "$MODELS_DIR" ] || [ -z "$(ls -A $MODELS_DIR/*.onnx 2>/dev/null)" ]; then
    echo "❌ Error: No ONNX models found in $MODELS_DIR"
    echo "Please add .onnx model files to $MODELS_DIR/ or set MODEL_URL to download one"
    exit 1
fi

# Get list of models
MODELS=($(ls "$MODELS_DIR"/*.onnx 2>/dev/null))
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
        echo "4 >>> Running Python $type Benchmark (CPU)..."
        uv run python/benchmarks/benchmark.py --model "$model_path" --input "$REPO_ROOT/$path" --device cpu --iterations "$ITERATIONS" --output "$RESULTS_DIR/python_cpu_${type}_${model_name}.json"
        
        # Python GPU
        echo "4 >>> Running Python $type Benchmark (GPU)..."
        uv run python/benchmarks/benchmark.py --model "$model_path" --input "$REPO_ROOT/$path" --device gpu --iterations "$ITERATIONS" --output "$RESULTS_DIR/python_gpu_${type}_${model_name}.json"
        
        # C++ CPU
        echo "4 >>> Running C++ $type Benchmark (CPU)..."
        "$REPO_ROOT/cpp/benchmarks/build/rfdetr_benchmark" "$model_path" "$REPO_ROOT/$path" cpu "$ITERATIONS"
        mv benchmark_cpp_cpu.json "$RESULTS_DIR/cpp_cpu_${type}_${model_name}.json"
        
        # C++ GPU
        echo "4 >>> Running C++ $type Benchmark (GPU)..."
        "$REPO_ROOT/cpp/benchmarks/build/rfdetr_benchmark" "$model_path" "$REPO_ROOT/$path" gpu "$ITERATIONS"
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

