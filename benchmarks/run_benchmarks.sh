#!/bin/bash
set -e

# Configuration
ITERATIONS=${1:-10}
MODEL_PATH="models/rf-detr-nano/rf-detr-nano.sim.onnx"
IMAGES_DIR="benchmarks/assets/images"
VIDEO_PATH="benchmarks/assets/video/sample.mp4"
ONNX_LIB_PATH="/home/essam/dev/libs/onnx/onnxruntime-linux-x64-gpu-1.21.0/lib"

# Setup environment
export LD_LIBRARY_PATH="$ONNX_LIB_PATH:$LD_LIBRARY_PATH"

# Paths
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_DIR="$REPO_ROOT/benchmarks"
RESULTS_DIR="$BENCH_DIR/results"

# Clean old results
echo ">>> Cleaning old results..."
rm -rf "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR"

echo ">>> Starting Benchmarks with $ITERATIONS iterations..."

# 0. Asset Check
if [ ! -d "$BENCH_DIR/assets" ]; then
    echo ">>> Downloading Benchmarking Assets..."
    mkdir -p "$BENCH_DIR/assets/images" "$BENCH_DIR/assets/video"
    curl -o "$BENCH_DIR/assets/images/coco_1.jpg" http://images.cocodataset.org/val2017/000000000139.jpg
    curl -o "$BENCH_DIR/assets/images/coco_2.jpg" http://images.cocodataset.org/val2017/000000000285.jpg
    curl -o "$BENCH_DIR/assets/images/coco_3.jpg" http://images.cocodataset.org/val2017/000000000632.jpg
    curl -o "$BENCH_DIR/assets/video/sample.mp4" https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/person-bicycle-car-detection.mp4
fi

# 1. Build C++ Library & Benchmark
echo ">>> Building C++ Library..."
cd "$REPO_ROOT/cpp"
mkdir -p build
cd build
cmake ..
make -j$(nproc)

echo ">>> Building C++ Benchmark..."
cd "$BENCH_DIR/cpp"
mkdir -p build
cd build
cmake ..
make -j$(nproc)

# 2. Run Python Benchmarks
cd "$REPO_ROOT/python"
test_inputs=("multi_images:$IMAGES_DIR" "video:$VIDEO_PATH")

for test in "${test_inputs[@]}"; do
    type="${test%%:*}"
    path="${test#*:}"
    echo ">>> Running Python $type Benchmark (CPU)..."
    uv run "../benchmarks/python/benchmark.py" --model "$REPO_ROOT/$MODEL_PATH" --input "$REPO_ROOT/$path" --device cpu --iterations "$ITERATIONS" --output "$RESULTS_DIR/python_cpu_${type}.json"
    
    echo ">>> Running Python $type Benchmark (GPU)..."
    uv run "../benchmarks/python/benchmark.py" --model "$REPO_ROOT/$MODEL_PATH" --input "$REPO_ROOT/$path" --device gpu --iterations "$ITERATIONS" --output "$RESULTS_DIR/python_gpu_${type}.json"
done

# 3. Run C++ Benchmarks
cd "$BENCH_DIR/cpp"
for test in "${test_inputs[@]}"; do
    type="${test%%:*}"
    path="${test#*:}"
    echo ">>> Running C++ $type Benchmark (CPU)..."
    ./build/rfdetr_benchmark "$REPO_ROOT/$MODEL_PATH" "$REPO_ROOT/$path" cpu "$ITERATIONS"
    mv benchmark_cpp_cpu.json "$RESULTS_DIR/cpp_cpu_${type}.json"
    
    echo ">>> Running C++ $type Benchmark (GPU)..."
    ./build/rfdetr_benchmark "$REPO_ROOT/$MODEL_PATH" "$REPO_ROOT/$path" gpu "$ITERATIONS"
    mv benchmark_cpp_gpu.json "$RESULTS_DIR/cpp_gpu_${type}.json"
done

# 4. Generate Report
echo ">>> Generating Results Report..."
cd "$BENCH_DIR"
uv run python3 -c "
import json, os
from datetime import datetime

results_dir = 'results'
output_md = 'results.md'

md = f'# RF-DETR Benchmark Results\n\nGenerated on: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}\n\n'

test_types = ['multi_images', 'video']
for t in test_types:
    md += f'## Test Case: {t.replace(\"_\", \" \").title()}\n\n'
    md += '| Implementation | Device | Prepro (ms) | ORT (ms) | Post (ms) | Total (ms) | FPS |\n'
    md += '| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n'
    
    rows = []
    for impl in ['python', 'cpp']:
        for dev in ['cpu', 'gpu']:
            f = f'{impl}_{dev}_{t}.json'
            path = os.path.join(results_dir, f)
            if os.path.exists(path):
                with open(path, 'r') as jf:
                    r = json.load(jf)
                    m = r['metrics']
                    rows.append({
                        'impl': r['implementation'],
                        'dev': r['device'].upper(),
                        'pre': m['preprocessing']['mean'],
                        'ort': m['ort_run']['mean'],
                        'post': m['postprocessing']['mean'],
                        'total': m['total_processing']['mean'],
                        'fps': m['total_processing']['fps']
                    })
    
    if rows:
        min_pre = min(r['pre'] for r in rows)
        min_ort = min(r['ort'] for r in rows)
        min_post = min(r['post'] for r in rows)
        min_total = min(r['total'] for r in rows)
        max_fps = max(r['fps'] for r in rows)
        
        for r in rows:
            pre_str = f'**{r[\"pre\"]:.2f}**' if r['pre'] == min_pre else f'{r[\"pre\"]:.2f}'
            ort_str = f'**{r[\"ort\"]:.2f}**' if r['ort'] == min_ort else f'{r[\"ort\"]:.2f}'
            post_str = f'**{r[\"post\"]:.2f}**' if r['post'] == min_post else f'{r[\"post\"]:.2f}'
            total_str = f'**{r[\"total\"]:.2f}**' if r['total'] == min_total else f'{r[\"total\"]:.2f}'
            fps_str = f'**{r[\"fps\"]:.2f}** 🚀' if r['fps'] == max_fps else f'{r[\"fps\"]:.2f}'
            md += f'| {r[\"impl\"]} | {r[\"dev\"]} | {pre_str} | {ort_str} | {post_str} | {total_str} | {fps_str} |\n'
    md += '\n'

with open(output_md, 'w') as out:
    out.write(md)

print(f'Done! Summary saved to {output_md}')
"

echo ">>> All Benchmarks Complete!"
