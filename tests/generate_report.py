import os
import json
import argparse
import numpy as np
from utils import match_detections

def parse_args():
    parser = argparse.ArgumentParser(description="Generate accuracy summary report.")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory containing results JSONs")
    parser.add_argument("--output", type=str, required=True, help="Path to save the Markdown report")
    return parser.parse_args()

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)

def calculate_metrics(ref_data, test_data):
    if not ref_data or not test_data:
        return 0, 0, "Error"
    
    ref_dets = ref_data["detections"]
    test_dets = test_data["detections"]
    
    matches, unmatched_ref, unmatched_test = match_detections(ref_dets, test_dets)
    
    avg_iou = np.mean([m[2] for m in matches]) if matches else 0
    recall = len(matches) / len(ref_dets) if ref_dets else 1.0
    
    status = "✅ PASS" if (avg_iou >= 0.9 and recall >= 0.8) else "❌ FAIL"
    if not ref_dets and not test_dets:
        status = "✅ PASS (No Dets)"
        avg_iou = 1.0

    return avg_iou, recall, status

def main():
    args = parse_args()
    
    ref_dir = os.path.join(args.results_dir, "reference")
    py_dir = os.path.join(args.results_dir, "python_onnx")
    cpp_dir = os.path.join(args.results_dir, "cpp_onnx")
    
    assets = [f for f in os.listdir(ref_dir) if f.endswith(".json")]
    
    report_lines = [
        "# Accuracy Validation Report",
        "",
        "| Asset | Implementation | Torch (ms) | ONNX (ms) | Speedup | Avg IOU | Status |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | :---: |"
    ]
    
    for asset_file in sorted(assets):
        asset_name = asset_file.replace(".json", "")
        ref_data = load_json(os.path.join(ref_dir, asset_file))
        py_data = load_json(os.path.join(py_dir, asset_file))
        cpp_data = load_json(os.path.join(cpp_dir, asset_file))
        
        torch_latency = ref_data.get('latency_ms', 0) if ref_data else 0
        
        # Python row
        if py_data:
            py_latency = py_data.get('latency_ms', 0)
            speedup = torch_latency / py_latency if py_latency > 0 else 0
            avg_iou, recall, status = calculate_metrics(ref_data, py_data)
            report_lines.append(f"| {asset_name} | Python ONNX | {torch_latency:.2f} | {py_latency:.2f} | {speedup:.1f}x | {avg_iou:.4f} | {status} |")
        
        # C++ row
        if cpp_data:
            cpp_latency = cpp_data.get('latency_ms', 0)
            speedup = torch_latency / cpp_latency if cpp_latency > 0 else 0
            avg_iou, recall, status = calculate_metrics(ref_data, cpp_data)
            report_lines.append(f"| | C++ ONNX | {torch_latency:.2f} | {cpp_latency:.2f} | {speedup:.1f}x | {avg_iou:.4f} | {status} |")
        
        report_lines.append("| | | | | | | |")

    with open(args.output, "w") as f:
        f.write("\n".join(report_lines))
    
    print(f"Report generated: {args.output}")

if __name__ == "__main__":
    main()
