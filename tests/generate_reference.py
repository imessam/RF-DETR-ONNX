import os
import sys
import time
import argparse
import json
import cv2
import torch
import numpy as np
from PIL import Image

# Add project root and python dir to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON_DIR = os.path.join(REPO_ROOT, "python")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

from rfdetr import RFDETRNano
from rfdetr.util.coco_classes import COCO_CLASSES


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Torch ground-truth reference results.")
    parser.add_argument("--weights", type=str, help="Path to Torch (.pth) weights")
    parser.add_argument("--assets_dir", type=str, required=True, help="Directory containing test images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save JSON results")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu"], help="Device to use")
    return parser.parse_args()

def save_detections_json(output_path, asset_name, boxes, labels, scores, latency_ms):
    detections = []
    for i in range(len(boxes)):
        class_id = int(labels[i])
        detections.append({
            "bbox": boxes[i].tolist(),
            "class_id": class_id,
            "class_name": COCO_CLASSES[class_id],
            "score": float(scores[i])
        })
    
    data = {
        "asset": asset_name,
        "implementation": "Torch (Reference)",
        "latency_ms": latency_ms,
        "detections": detections
    }
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = "cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu"
    print(f"Initializing Torch Reference Model on {device}...")
    
    # Initialize model
    model_kwargs = {}
    if args.weights:
        model_kwargs["pretrain_weights"] = args.weights
    
    model = RFDETRNano(device=device, **model_kwargs)
    model.optimize_for_inference()
    
    assets = [f for f in os.listdir(args.assets_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not assets:
        print(f"No images found in {args.assets_dir}")
        return

    print(f"Generating reference results for {len(assets)} assets...")
    for asset_name in assets:
        asset_path = os.path.join(args.assets_dir, asset_name)
        image_bgr = cv2.imread(asset_path)
        if image_bgr is None:
            continue
            
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        
        start_time = time.perf_counter()
        res = model.predict(image_pil, threshold=args.threshold)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = (time.perf_counter() - start_time) * 1000
        
        base_name = os.path.splitext(asset_name)[0]
        output_json = os.path.join(args.output_dir, f"{base_name}.json")
        save_detections_json(output_json, asset_name, res.xyxy, res.class_id, res.confidence, latency)
        print(f"  - {asset_name}: {len(res.xyxy)} detections ({latency:.2f} ms)")

if __name__ == "__main__":
    main()
