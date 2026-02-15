import os
import json
import numpy as np
from rfdetr.util.coco_classes import COCO_CLASSES

class AccuracyReporter:
    """Helper class to handle validation report generation."""

    def __init__(self, config):
        self.config = config
        self.results = []
        os.makedirs(config["results_dir"], exist_ok=True)

    def add_result(self, asset_result):
        """Add a result for an asset."""
        self.results.append(asset_result)

    def save_detections_json(self, asset_name, suffix, boxes, labels, scores, latency=0):
        """Save detection details to a JSON file."""
        base_name = os.path.splitext(asset_name)[0]
        output_path = os.path.join(self.config["results_dir"], f"{base_name}_{suffix}.json")
        
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
            
        data = {
            "asset": asset_name,
            "latency_ms": latency,
            "detections": detections
        }
            
        with open(output_path, "w") as f:
            json.dump(data, f, indent=4)

    def load_detections_json(self, asset_name, suffix):
        """Load detection details from a JSON file."""
        base_name = os.path.splitext(asset_name)[0]
        input_path = os.path.join(self.config["results_dir"], f"{base_name}_{suffix}.json")
        
        if not os.path.exists(input_path):
            return None
            
        with open(input_path, "r") as f:
            return json.load(f)

    def generate_reports(self):
        """Generate final JSON and Markdown reports."""
        if not self.results:
            return
            
        summary_data = {
            "config": self.config,
            "results": self.results,
            "average_iou": float(np.mean([r['avg_iou'] for r in self.results])),
            "average_speedup": float(np.mean([r['pt_time']/r['ox_time'] for r in self.results])) if all(r['ox_time'] > 0 for r in self.results) else 0
        }

        # 1. Save JSON summary
        report_path_json = os.path.join(self.config["results_dir"], "summary_report.json")
        with open(report_path_json, "w") as f:
            json.dump(summary_data, f, indent=4)
        print(f"\n--- Validation Report Saved to {report_path_json} ---")

        # 2. Save Markdown summary
        self._save_markdown_summary(summary_data)

    def _save_markdown_summary(self, summary_data):
        """Generate a human-readable markdown report."""
        report_path_md = os.path.join(self.config["results_dir"], "summary_report.md")
        
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
            speedup = res['pt_time'] / res['ox_time'] if res['ox_time'] > 0 else 0
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
                f"- **Speedup**: {res['pt_time'] / res['ox_time']:.2f}x" if res['ox_time'] > 0 else f"- **Speedup**: N/A",
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
