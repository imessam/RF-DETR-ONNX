import sys
import os
import argparse
import time
import numpy as np
import cv2
import json

# Add project root to sys.path
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if base_path not in sys.path:
    sys.path.insert(0, os.path.join(base_path, 'python'))

from modules.model import RFDETRModel

def run_benchmark(model_path, data_path, device, num_iterations=100, warmup_iterations=10):
    print(f"Initializing model: {model_path} on {device}")
    model = RFDETRModel(model_path, device=device)
    
    # Identify if data_path is directory, video, or image
    is_dir = os.path.isdir(data_path)
    is_video = data_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    
    images = []
    if is_dir:
        for f in os.listdir(data_path):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                images.append(cv2.imread(os.path.join(data_path, f)))
        if not images:
            print(f"Error: No images found in {data_path}")
            sys.exit(1)
        print(f"Loaded {len(images)} images from directory.")
    elif is_video:
        cap = cv2.VideoCapture(data_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {data_path}")
            sys.exit(1)
        # Read a few frames for warmup and benchmarking
        count = 0
        while count < num_iterations + warmup_iterations:
            ret, frame = cap.read()
            if not ret: break
            images.append(frame)
            count += 1
        cap.release()
        print(f"Loaded {len(images)} frames from video.")
    else:
        img = cv2.imread(data_path)
        if img is None:
            print(f"Error: Could not load image {data_path}")
            sys.exit(1)
        images = [img]

    print(f"Running {warmup_iterations} warmup iterations...")
    for i in range(warmup_iterations):
        model.predict(images[i % len(images)])

    print(f"Running {num_iterations} benchmark iterations...")
    
    pre_times = []
    ort_times = []
    post_times = []
    total_times = []

    for i in range(num_iterations):
        img_idx = i % len(images)
        _, _, _, _, timings = model.predict(images[img_idx])
        pre_times.append(timings['preprocess'])
        ort_times.append(timings['ort_run'])
        post_times.append(timings['postprocess'])
        total_times.append(timings['preprocess'] + timings['ort_run'] + timings['postprocess'])
        
        if (i + 1) % 10 == 0:
            print(f"Iteration {i+1}/{num_iterations}")

    results = {
        "implementation": "Python",
        "device": device,
        "num_iterations": num_iterations,
        "data_source": data_path,
        "metrics": {
            "preprocessing": {
                "mean": np.mean(pre_times),
                "std": np.std(pre_times),
                "min": np.min(pre_times),
                "max": np.max(pre_times)
            },
            "ort_run": {
                "mean": np.mean(ort_times),
                "std": np.std(ort_times),
                "min": np.min(ort_times),
                "max": np.max(ort_times)
            },
            "postprocessing": {
                "mean": np.mean(post_times),
                "std": np.std(post_times),
                "min": np.min(post_times),
                "max": np.max(post_times)
            },
            "total_processing": {
                "mean": np.mean(total_times),
                "std": np.std(total_times),
                "min": np.min(total_times),
                "max": np.max(total_times),
                "fps": 1000.0 / np.mean(total_times)
            }
        }
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Python Benchmark for RF-DETR ONNX")
    parser.add_argument("--model", required=True, type=str, help="Path to ONNX model")
    parser.add_argument("--input", required=True, type=str, help="Path to input image, directory, or video")
    parser.add_argument("--device", default="gpu", choices=["gpu", "cpu"], help="Device to use")
    parser.add_argument("--iterations", default=100, type=int, help="Number of iterations")
    parser.add_argument("--output", default="benchmark_python.json", type=str, help="Output JSON file")
    
    args = parser.parse_args()
    
    results = run_benchmark(args.model, args.input, args.device, args.iterations)
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
