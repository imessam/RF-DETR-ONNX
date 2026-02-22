import os
import sys
import time
import argparse
import json
import cv2
import numpy as np

# Add project root and python dir to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PYTHON_DIR = os.path.join(REPO_ROOT, "python")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

from modules.model import RFDETRModel
from rfdetr.util.coco_classes import COCO_CLASSES

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Python ONNX inference results.")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--input", type=str, required=True, help="Directory containing test images")
    parser.add_argument("--output", type=str, required=True, help="Directory to save JSON results")
    parser.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu"], help="Device to use")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()

def save_detections_json(output_path, asset_name, detections, latency_ms):
    results = []
    for det in detections:
        results.append({
            "bbox": det.normalized_box.tolist(), # [x, y, w, h] normalized
            "class_id": det.label,
            "class_name": COCO_CLASSES[det.label],
            "score": float(det.score)
        })
    
    data = {
        "asset": asset_name,
        "implementation": "Python ONNX",
        "latency_ms": latency_ms,
        "detections": results
    }
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Initializing Python ONNX Model on {args.device}...")
    model = RFDETRModel(args.model, device=args.device)
    
    assets = [f for f in os.listdir(args.input) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not assets:
        print(f"No images found in {args.input}")
        return

    print(f"Generating Python ONNX results for {len(assets)} assets...")
    for asset_name in assets:
        asset_path = os.path.join(args.input, asset_name)
        image_bgr = cv2.imread(asset_path)
        if image_bgr is None:
            continue
            
        start_time = time.perf_counter()
        detections, timings = model.predict(image_bgr, confidence_threshold=args.threshold)
        latency = (time.perf_counter() - start_time) * 1000
        
        base_name = os.path.splitext(asset_name)[0]
        output_json = os.path.join(args.output, f"{base_name}.json")
        save_detections_json(output_json, asset_name, detections, latency)
        
        if args.verbose:
            print(f"  - {asset_name}: {len(detections)} detections ({latency:.2f} ms)")

if __name__ == "__main__":
    main()
