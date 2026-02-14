import os
import time
import json
import numpy as np
import torch
import pytest
from PIL import Image
from rfdetr import RFDETRNano, RFDETRSmall, RFDETRBase, RFDETRMedium
from modules.model import RFDETRModel

# Architecture mapping
MODELS = {
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "base": RFDETRBase,
    "medium": RFDETRMedium,
}

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

class TestValidation:
    _results = []
    _config = None

    @pytest.fixture(autouse=True, scope="class")
    def setup_class(self, test_config):
        """Initialize models once for the test class."""
        TestValidation._config = test_config
        os.makedirs(test_config["results_dir"], exist_ok=True)
        
        device = "cuda" if test_config["device"] == "gpu" and torch.cuda.is_available() else "cpu"
        
        # Initialize models
        torch_model = MODELS[test_config["arch"]](device=device)
        torch_model.optimize_for_inference()
        onnx_model = RFDETRModel(test_config["model_path"], device=test_config["device"])
        
        yield torch_model, onnx_model
        
        # Final reporting after all tests in class
        self._save_summary(test_config)

    def _save_summary(self, config):
        if not TestValidation._results:
            return
            
        report_path = os.path.join(config["results_dir"], "summary_report.json")
        with open(report_path, "w") as f:
            json.dump({
                "config": config,
                "results": TestValidation._results,
                "average_iou": float(np.mean([r['avg_iou'] for r in TestValidation._results])),
                "average_speedup": float(np.mean([r['pt_time']/r['ox_time'] for r in TestValidation._results]))
            }, f, indent=4)
        print(f"\n--- Validation Report Saved to {report_path} ---")

    @pytest.mark.parametrize("asset_name", [
        f for f in os.listdir("tests/test_assets") 
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    def test_detection_accuracy(self, setup_class, asset_name):
        """Standardized Pytest for accuracy validation."""
        torch_model, onnx_model = setup_class
        config = TestValidation._config
        asset_path = os.path.join(config["assets_dir"], asset_name)
        
        image = Image.open(asset_path).convert("RGB")
        
        # PyTorch Inference
        start_pt = time.perf_counter()
        pt_res = torch_model.predict(image, threshold=config["threshold"])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        pt_time = (time.perf_counter() - start_pt) * 1000

        # ONNX Inference
        start_ox = time.perf_counter()
        ox_scores, ox_labels, ox_boxes, ox_masks, _ = onnx_model.predict(asset_path, config["threshold"])
        ox_time = (time.perf_counter() - start_ox) * 1000

        # Comparison metrics
        k = min(len(pt_res.xyxy), len(ox_boxes), 5)
        ious = []
        label_matches = 0
        
        for i in range(k):
            iou = calculate_iou(pt_res.xyxy[i], ox_boxes[i])
            ious.append(iou)
            if pt_res.class_id[i] == ox_labels[i]:
                label_matches += 1
        
        avg_iou = np.mean(ious) if ious else 1.0
        label_acc = (label_matches / k) * 100 if k > 0 else 100.0
        
        asset_result = {
            "asset": asset_name,
            "pt_time": float(pt_time),
            "ox_time": float(ox_time),
            "avg_iou": float(avg_iou),
            "label_acc": float(label_acc),
            "status": "PASS" if avg_iou > 0.9 and label_acc == 100 else "FAIL"
        }
        TestValidation._results.append(asset_result)
        
        # Save annotated image
        output_path = os.path.join(config["results_dir"], f"res_{asset_name}")
        onnx_model.save_detections(asset_path, ox_boxes, ox_labels, ox_masks, output_path)

        # Assertions for Pytest
        assert pt_res.class_id[:k].tolist() == ox_labels[:k].tolist(), f"Labels mismatch for {asset_name}"
        assert avg_iou > 0.9, f"Average IOU {avg_iou:.4f} below threshold for {asset_name}"
