import sys
import os 

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if base_path not in sys.path:
    sys.path.insert(0, base_path) 

import time
import json
import numpy as np
import torch
import pytest
import cv2
from PIL import Image
from rfdetr import RFDETRNano, RFDETRSmall, RFDETRBase, RFDETRMedium
from rfdetr.util.coco_classes import COCO_CLASSES
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
        if not os.path.exists(test_config["model_path"]):
            pytest.skip(f"Model not found at {test_config['model_path']}")
            
        torch_model = MODELS[test_config["arch"]](device=device)
        torch_model.optimize_for_inference()
        onnx_model = RFDETRModel(test_config["model_path"], device=test_config["device"])
        
        yield torch_model, onnx_model
        
        # Final reporting after all tests in class
        self._save_summary(test_config)

    def _save_summary(self, config):
        if not TestValidation._results:
            return
            
        # 1. Save JSON summary
        report_path_json = os.path.join(config["results_dir"], "summary_report.json")
        summary_data = {
            "config": config,
            "results": TestValidation._results,
            "average_iou": float(np.mean([r['avg_iou'] for r in TestValidation._results])),
            "average_speedup": float(np.mean([r['pt_time']/r['ox_time'] for r in TestValidation._results]))
        }
        with open(report_path_json, "w") as f:
            json.dump(summary_data, f, indent=4)
        print(f"\n--- Validation Report Saved to {report_path_json} ---")

        # 2. Save Markdown summary
        self._save_markdown_summary(config, summary_data)

    def _save_markdown_summary(self, config, summary_data):
        """Generate a human-readable markdown report."""
        report_path_md = os.path.join(config["results_dir"], "summary_report.md")
        
        lines = [
            "# RF-DETR Validation Report",
            "",
            "## Summary Statistics",
            f"- **Average IOU**: {summary_data['average_iou']:.4f}",
            f"- **Average Speedup (ONNX/Torch)**: {summary_data['average_speedup']:.2f}x",
            "",
            "## Assets Table",
            "| Asset | Torch (ms) | ONNX (ms) | Speedup | Avg IOU | Label Acc | Status |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
        ]
        
        for res in summary_data["results"]:
            speedup = res['pt_time'] / res['ox_time']
            status_emoji = "✅ PASS" if res['status'] == "PASS" else "❌ FAIL"
            lines.append(
                f"| {res['asset']} | {res['pt_time']:.2f} | {res['ox_time']:.2f} | {speedup:.2f}x | "
                f"{res['avg_iou']:.4f} | {res['label_acc']:.1f}% | {status_emoji} |"
            )
            
        lines.append("")
        lines.append("## Detailed Results")
        
        for res in summary_data["results"]:
            base_name = os.path.splitext(res['asset'])[0]
            lines.extend([
                f"### {res['asset']}",
                f"- **Status**: {res['status']}",
                f"- **Avg IOU**: {res['avg_iou']:.4f}",
                f"- **Speedup**: {res['pt_time'] / res['ox_time']:.2f}x",
                "",
                "#### Bounding Box Comparison (Greedy Matching)",
                "We use a greedy matching algorithm (IOU > 0.4) to pair detections from PyTorch and ONNX models.",
                "",
                "| Match # | Torch Class | ONNX Class | IOU | Status |",
                "| :--- | :--- | :--- | :---: | :---: |"
            ])
            
            for i, match in enumerate(res.get('box_matches', [])):
                match_status = "✅ OK" if match['iou'] > 0.9 and match['pt_label'] == match['ox_label'] else "⚠️ DIFF"
                lines.append(
                    f"| {i+1} | {match['pt_label']} | {match['ox_label']} | {match['iou']:.4f} | {match_status} |"
                )
            
            lines.extend([
                "",
                "| Torch Annotated | ONNX Annotated |",
                "| :---: | :---: |",
                f"| ![{res['asset']} Torch]({base_name}_torch_annotated.jpg) | ![{res['asset']} ONNX]({base_name}_onnx_annotated.jpg) |",
                ""
            ])
            
        with open(report_path_md, "w") as f:
            f.write("\n".join(lines))
        print(f"--- Markdown Report Saved to {report_path_md} ---")

    def _save_detections_json(self, asset_name, suffix, boxes, labels, scores, masks=None):
        """Save detection details to a JSON file."""
        config = TestValidation._config
        base_name = os.path.splitext(asset_name)[0]
        output_path = os.path.join(config["results_dir"], f"{base_name}_{suffix}.json")
        
        detections = []
        for i in range(len(boxes)):
            class_id = int(labels[i])
            det = {
                "bbox": boxes[i].tolist(),
                "class_id": class_id,
                "class_name": COCO_CLASSES[class_id],
                "score": float(scores[i])
            }
            detections.append(det)
            
        with open(output_path, "w") as f:
            json.dump(detections, f, indent=4)

    @pytest.mark.parametrize("asset_name", [
        f for f in (os.listdir("tests/test_assets") if os.path.exists("tests/test_assets") else [])
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    def test_detection_accuracy(self, setup_class, asset_name):
        """Standardized Pytest for accuracy validation."""
        torch_model, onnx_model = setup_class
        config = TestValidation._config
        asset_path = os.path.join(config["assets_dir"], asset_name)
        
        image_bgr = cv2.imread(asset_path)
        if image_bgr is None:
            pytest.skip(f"Could not load asset: {asset_path}")
        
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        
        # PyTorch Inference
        start_pt = time.perf_counter()
        pt_res = torch_model.predict(image_pil, threshold=config["threshold"])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        pt_time = (time.perf_counter() - start_pt) * 1000

        # ONNX Inference
        start_ox = time.perf_counter()
        ox_scores, ox_labels, ox_boxes, ox_masks, _ = onnx_model.predict(image_bgr, config["threshold"])
        ox_time = (time.perf_counter() - start_ox) * 1000

        # 1. Save individual detection results to JSON
        self._save_detections_json(asset_name, "torch", pt_res.xyxy, pt_res.class_id, pt_res.confidence)
        self._save_detections_json(asset_name, "onnx", ox_boxes, ox_labels, ox_scores)

        # 2. Save annotated images for both
        base_name = os.path.splitext(asset_name)[0]
        torch_img_path = os.path.join(config["results_dir"], f"{base_name}_torch_annotated.jpg")
        onnx_img_path = os.path.join(config["results_dir"], f"{base_name}_onnx_annotated.jpg")
        
        pt_labels_str = [COCO_CLASSES[int(lid)] for lid in pt_res.class_id]
        ox_labels_str = [COCO_CLASSES[int(lid)] for lid in ox_labels]
        
        onnx_model.save_detections(image_bgr, pt_res.xyxy, pt_labels_str, None, torch_img_path)
        onnx_model.save_detections(image_bgr, ox_boxes, ox_labels_str, ox_masks, onnx_img_path)

        # Comparison metrics with greedy matching
        ious = []
        box_matches = []
        label_matches = 0
        matched_ox_indices = set()
        
        num_pt = len(pt_res.xyxy)
        num_ox = len(ox_boxes)
        
        for i in range(num_pt):
            best_iou = 0
            best_idx = -1
            for j in range(num_ox):
                if j in matched_ox_indices:
                    continue
                iou = calculate_iou(pt_res.xyxy[i], ox_boxes[j])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j
            
            if best_idx != -1 and best_iou > 0.4:
                ious.append(best_iou)
                matched_ox_indices.add(best_idx)
                is_label_match = (pt_res.class_id[i] == ox_labels[best_idx])
                if is_label_match:
                    label_matches += 1
                
                box_matches.append({
                    "pt_label": COCO_CLASSES[int(pt_res.class_id[i])],
                    "ox_label": COCO_CLASSES[int(ox_labels[best_idx])],
                    "iou": float(best_iou)
                })
        
        k = len(ious)
        avg_iou = np.mean(ious) if ious else (1.0 if num_pt == 0 and num_ox == 0 else 0.0)
        label_acc = (label_matches / k) * 100 if k > 0 else (100.0 if num_pt == 0 else 0.0)
        asset_result = {
            "asset": asset_name,
            "pt_time": float(pt_time),
            "ox_time": float(ox_time),
            "avg_iou": float(avg_iou),
            "label_acc": float(label_acc),
            "box_matches": box_matches,
            "status": "PASS" if avg_iou > 0.9 and label_acc == 100 else "FAIL"
        }
        TestValidation._results.append(asset_result)
        
        assert label_acc == 100, f"Labels mismatch for {asset_name}: {label_matches}/{num_pt} matched"
        assert avg_iou > 0.9, f"Average IOU {avg_iou:.4f} below threshold for {asset_name}"
