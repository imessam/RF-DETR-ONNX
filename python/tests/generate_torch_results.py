import os
import sys
import time

# Add the project root to sys.path
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if base_path not in sys.path:
    sys.path.insert(0, base_path)
import argparse
import cv2
import torch
import numpy as np
from PIL import Image
from rfdetr import RFDETRNano, RFDETRSmall, RFDETRBase, RFDETRMedium
from rfdetr.util.coco_classes import COCO_CLASSES
from tests.reporting import AccuracyReporter

# Architecture mapping
MODELS = {
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "base": RFDETRBase,
    "medium": RFDETRMedium,
}

def parse_args():
    parser = argparse.ArgumentParser(description="Generate PyTorch inference results for accuracy validation.")
    parser.add_argument("--arch", type=str, default="nano", choices=MODELS.keys(), help="Model architecture")
    parser.add_argument("--weights", type=str, help="Optional path to custom weights")
    parser.add_argument("--assets_dir", type=str, default="tests/test_assets", help="Directory containing test assets")
    parser.add_argument("--results_dir", type=str, default="tests/test_results", help="Directory to save results")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="gpu", choices=["cpu", "gpu"], help="Device to use")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    
    device = "cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize model
    model_kwargs = {}
    if args.weights:
        model_kwargs["pretrain_weights"] = args.weights
    
    model = MODELS[args.arch](device=device, **model_kwargs)
    model.optimize_for_inference()
    
    # Initialize reporter (reusing it to save detections)
    # Note: Reporter expects a config dict with 'results_dir'
    reporter = AccuracyReporter({"results_dir": args.results_dir})
    
    # Get assets
    assets = [f for f in os.listdir(args.assets_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not assets:
        print(f"No image assets found in {args.assets_dir}")
        return

    from modules.model import RFDETRModel # For save_detections helper

    for asset_name in assets:
        asset_path = os.path.join(args.assets_dir, asset_name)
        print(f"Processing {asset_name}...")
        
        image_bgr = cv2.imread(asset_path)
        if image_bgr is None:
            print(f"Failed to load {asset_path}")
            continue
            
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        
        # Inference
        start_time = time.perf_counter()
        res = model.predict(image_pil, threshold=args.threshold)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = (time.perf_counter() - start_time) * 1000
        
        # Save JSON
        reporter.save_detections_json(asset_name, "torch", res.xyxy, res.class_id, res.confidence, latency=latency)
        
        # Save Annotated Image
        base_name = os.path.splitext(asset_name)[0]
        output_img_path = os.path.join(args.results_dir, f"{base_name}_torch_annotated.jpg")
        labels_str = [COCO_CLASSES[int(lid)] for lid in res.class_id]
        
        # We use RFDETRModel.save_detections as a utility if possible, 
        # or we just need a dummy instance if it's not a static method.
        # Actually, let's just use the logic directly or call it via a dummy.
        dummy_model = RFDETRModel.__new__(RFDETRModel)
        dummy_model.save_detections(image_bgr, res.xyxy, res.class_id, None, output_img_path)
        
        print(f"Done. Latency: {latency:.2f} ms")

if __name__ == "__main__":
    main()
