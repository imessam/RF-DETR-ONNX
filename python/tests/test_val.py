import os
import numpy as np
import pytest
from tests.reporting import AccuracyReporter

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

class TestAccuracy:
    """Compare pre-generated PyTorch and ONNX inference results."""
    
    @pytest.fixture(autouse=True, scope="class")
    def reporter(self, test_config):
        """Fixture to initialize the reporter for the test class."""
        reporter = AccuracyReporter(test_config)
        yield reporter
        reporter.generate_reports()

    @pytest.mark.parametrize("asset_name", [
        f for f in (os.listdir("tests/test_assets") if os.path.exists("tests/test_assets") else [])
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    def test_compare_detections(self, reporter, test_config, asset_name):
        """Compare Torch and ONNX detections for a single asset."""
        torch_data = reporter.load_detections_json(asset_name, "torch")
        onnx_data = reporter.load_detections_json(asset_name, "onnx")
        
        if torch_data is None:
            pytest.fail(f"Torch results missing for {asset_name}. Run generate_torch_results.py first.")
        if onnx_data is None:
            pytest.fail(f"ONNX results missing for {asset_name}. Run generate_onnx_results.py first.")
            
        pt_dets = torch_data["detections"]
        ox_dets = onnx_data["detections"]
        
        pt_boxes = np.array([d["bbox"] for d in pt_dets]) if pt_dets else np.empty((0, 4))
        pt_labels = np.array([d["class_id"] for d in pt_dets])
        
        ox_boxes = np.array([d["bbox"] for d in ox_dets]) if ox_dets else np.empty((0, 4))
        ox_labels = np.array([d["class_id"] for d in ox_dets])
        
        # Comparison metrics with greedy matching
        ious = []
        box_matches = []
        label_matches = 0
        matched_ox_indices = set()
        
        num_pt = len(pt_boxes)
        num_ox = len(ox_boxes)
        
        for i in range(num_pt):
            best_iou = 0
            best_idx = -1
            for j in range(num_ox):
                if j in matched_ox_indices:
                    continue
                iou = calculate_iou(pt_boxes[i], ox_boxes[j])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j
            
            if best_idx != -1 and best_iou > 0.4:
                ious.append(best_iou)
                matched_ox_indices.add(best_idx)
                is_label_match = (pt_labels[i] == ox_labels[best_idx])
                if is_label_match:
                    label_matches += 1
                
                box_matches.append({
                    "pt_label": pt_dets[i]["class_name"],
                    "ox_label": ox_dets[best_idx]["class_name"],
                    "iou": float(best_iou)
                })
        
        k = len(ious)
        avg_iou = np.mean(ious) if ious else (1.0 if num_pt == 0 and num_ox == 0 else 0.0)
        label_acc = (label_matches / k) * 100 if k > 0 else (100.0 if num_pt == 0 else 0.0)
        
        asset_result = {
            "asset": asset_name,
            "pt_time": float(torch_data.get("latency_ms", 0)),
            "ox_time": float(onnx_data.get("latency_ms", 0)),
            "avg_iou": float(avg_iou),
            "label_acc": float(label_acc),
            "box_matches": box_matches,
            "status": "PASS" if avg_iou > 0.9 and label_acc == 100 else "FAIL"
        }
        reporter.add_result(asset_result)
        
        assert label_acc == 100, f"Labels mismatch for {asset_name}: {label_matches}/{num_pt} matched"
        assert avg_iou > 0.9, f"Average IOU {avg_iou:.4f} below threshold for {asset_name}"
