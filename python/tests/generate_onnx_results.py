import os
import sys
import time

# Add the project root to sys.path
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if base_path not in sys.path:
    sys.path.insert(0, base_path)

import argparse
import cv2
import numpy as np
from modules.model import RFDETRModel, DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_MAX_NUMBER_BOXES
from rfdetr.util.coco_classes import COCO_CLASSES
from tests.reporting import AccuracyReporter

def parse_args():
    parser = argparse.ArgumentParser(description="Generate ONNX inference results for accuracy validation.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu"], help="Device to use")
    parser.add_argument("--assets_dir", type=str, default="tests/test_assets", help="Directory containing test assets")
    parser.add_argument("--results_dir", type=str, default="tests/test_results", help="Directory to save results")
    parser.add_argument("--threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD, help="Confidence threshold")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Initialize model
    model = RFDETRModel(args.model_path, device=args.device)
    
    # Initialize reporter
    reporter = AccuracyReporter({"results_dir": args.results_dir})
    
    # Get assets
    assets = [f for f in os.listdir(args.assets_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not assets:
        print(f"No image assets found in {args.assets_dir}")
        return

    for asset_name in assets:
        asset_path = os.path.join(args.assets_dir, asset_name)
        print(f"Processing {asset_name}...")
        
        image_bgr = cv2.imread(asset_path)
        if image_bgr is None:
            print(f"Failed to load {asset_path}")
            continue
            
        # Inference
        start_time = time.perf_counter()
        scores, labels, boxes, masks, timings = model.predict(image_bgr, args.threshold)
        latency = (time.perf_counter() - start_time) * 1000
        
        # Save JSON
        reporter.save_detections_json(asset_name, "onnx", boxes, labels, scores, latency=latency)
        
        # Save Annotated Image
        base_name = os.path.splitext(asset_name)[0]
        output_img_path = os.path.join(args.results_dir, f"{base_name}_onnx_annotated.jpg")
        
        model.save_detections(image_bgr, boxes, labels, masks, output_img_path)
        
        print(f"Done. Latency: {latency:.2f} ms (ORT: {timings['ort_run']:.2f} ms)")

if __name__ == "__main__":
    main()
