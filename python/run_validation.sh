#!/bin/bash
set -e

# Configuration
ARCH=${1:-nano}
MODEL_PATH="tests/test_models/inference_model.sim.onnx"
ASSETS_DIR="tests/test_assets"
RESULTS_DIR="tests/test_results"
THRESHOLD=0.5

echo "--- RF-DETR Validation Pipeline ---"
echo "Architecture: $ARCH"
echo "Model Path:   $MODEL_PATH"
echo "----------------------------------"

# 0. Sync dependencies (ensure all extras are installed)
echo "[0/4] Syncing dependencies (extras: test, export)..."
uv sync --extra test --extra export

# 1. Preparation (Download and Export)
echo "[1/4] Preparing model (download and export to ONNX)..."
uv run tests/prepare_models.py

# 2. Generate PyTorch (Reference) Results
echo "[2/4] Generating PyTorch reference results..."
uv run tests/generate_torch_results.py --arch "$ARCH" --assets_dir "$ASSETS_DIR" --results_dir "$RESULTS_DIR" --threshold "$THRESHOLD"

# 3. Generate ONNX (Target) Results
echo "[3/4] Generating ONNX target results..."
uv run tests/generate_onnx_results.py --model_path "$MODEL_PATH" --assets_dir "$ASSETS_DIR" --results_dir "$RESULTS_DIR" --threshold "$THRESHOLD"

# 4. Run Comparison and Generate Report
echo "[4/4] Running accuracy comparison..."
uv run pytest tests/test_val.py --arch "$ARCH" --model_path "$MODEL_PATH" --assets_dir "$ASSETS_DIR" --results_dir "$RESULTS_DIR" --threshold "$THRESHOLD"

echo "----------------------------------"
echo "Validation complete. See report at: $RESULTS_DIR/summary_report.md"
