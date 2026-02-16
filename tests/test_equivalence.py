import os
import json
import pytest
import numpy as np
from utils import calculate_iou, match_detections

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)

def get_assets(directory):
    if not directory or not os.path.exists(directory):
        return []
    return [f for f in os.listdir(directory) if f.endswith(".json")]

def compare_results(ref_data, test_data, label):
    if not ref_data or not test_data:
        pytest.fail(f"Missing data for {label}")
        
    ref_dets = ref_data["detections"]
    test_dets = test_data["detections"]
    
    matches, unmatched_ref, unmatched_test = match_detections(ref_dets, test_dets)
    
    print(f"\nComparing {label} vs Reference for {ref_data['asset']}:")
    print(f"  Reference detections: {len(ref_dets)}")
    print(f"  {label} detections: {len(test_dets)}")
    print(f"  Matches: {len(matches)}")
    
    # Assertions
    # 1. Most reference detections should be found
    if len(ref_dets) > 0:
        recall = len(matches) / len(ref_dets)
        assert recall >= 0.8, f"Recall too low: {recall:.2f}. Found {len(matches)}/{len(ref_dets)}"
        
    # 2. Average IoU of matches should be high
    if matches:
        avg_iou = np.mean([m[2] for m in matches])
        assert avg_iou >= 0.9, f"Average IoU too low: {avg_iou:.4f} for {label}"

def test_python_vs_reference(ref_dir, py_dir):
    assets = get_assets(ref_dir)
    assert assets, "No reference JSONs found"
    
    for asset in assets:
        ref_data = load_json(os.path.join(ref_dir, asset))
        py_data = load_json(os.path.join(py_dir, asset))
        compare_results(ref_data, py_data, "Python ONNX")

def test_cpp_vs_reference(ref_dir, cpp_dir):
    assets = get_assets(ref_dir)
    assert assets, "No reference JSONs found"
    
    for asset in assets:
        ref_data = load_json(os.path.join(ref_dir, asset))
        cpp_data = load_json(os.path.join(cpp_dir, asset))
        compare_results(ref_data, cpp_data, "C++ ONNX")

def test_python_vs_cpp(py_dir, cpp_dir):
    assets = get_assets(py_dir)
    assert assets, "No Python JSONs found"
    
    for asset in assets:
        py_data = load_json(os.path.join(py_dir, asset))
        cpp_data = load_json(os.path.join(cpp_dir, asset))
        compare_results(py_data, cpp_data, "Python vs C++ ONNX")
