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

def run_benchmark(model_path, data_path, device, num_iterations=100, warmup_iterations=10, sleep_per_image=0.0, verbose=False):
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
        if sleep_per_image > 0:
            time.sleep(sleep_per_image)
        img_idx = i % len(images)
        _, _, _, _, timings = model.predict(images[img_idx])
        
        pre = timings['preprocess']
        ort = timings['ort_run']
        post = timings['postprocess']
        total = pre + ort + post
        
        pre_times.append(pre)
        ort_times.append(ort)
        post_times.append(post)
        total_times.append(total)
        
        if verbose:
            print(f"Iteration {i+1:3d}: Pre: {pre:6.2f}ms, ORT: {ort:6.2f}ms, Post: {post:6.2f}ms, Total: {total:6.2f}ms")
        elif (i + 1) % 10 == 0:
            print(f"Iteration {i+1}/{num_iterations}")

    results = {
        "implementation": "Python",
        "device": device,
        "num_iterations": num_iterations,
        "data_source": data_path,
        "metrics": {
            "preprocessing": {
                "mean": np.mean(pre_times),
                "median": np.median(pre_times),
                "std": np.std(pre_times),
                "min": np.min(pre_times),
                "max": np.max(pre_times)
            },
            "ort_run": {
                "mean": np.mean(ort_times),
                "median": np.median(ort_times),
                "std": np.std(ort_times),
                "min": np.min(ort_times),
                "max": np.max(ort_times)
            },
            "postprocessing": {
                "mean": np.mean(post_times),
                "median": np.median(post_times),
                "std": np.std(post_times),
                "min": np.min(post_times),
                "max": np.max(post_times)
            },
            "total_processing": {
                "mean": np.mean(total_times),
                "median": np.median(total_times),
                "std": np.std(total_times),
                "min": np.min(total_times),
                "max": np.max(total_times),
                "fps": 1000.0 / np.median(total_times)
            }
        },
        "iterations": [
            {
                "index": i + 1,
                "preprocessing": pre_times[i],
                "ort_run": ort_times[i],
                "postprocessing": post_times[i],
                "total": total_times[i]
            } for i in range(num_iterations)
        ]
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Python Benchmark for RF-DETR ONNX")
    parser.add_argument("--model", required=True, type=str, help="Path to ONNX model")
    parser.add_argument("--input", required=True, type=str, help="Path to input image, directory, or video")
    parser.add_argument("--device", default="gpu", choices=["gpu", "cpu"], help="Device to use")
    parser.add_argument("--iterations", default=100, type=int, help="Number of iterations")
    parser.add_argument("--sleep-per-image", default=0.0, type=float, help="Seconds to sleep between images (default: 0.0)")
    parser.add_argument("--verbose", action="store_true", help="Enable per-iteration logging")
    parser.add_argument("--output", default="benchmark_python.json", type=str, help="Output JSON file")
    
    args = parser.parse_args()
    
    results = run_benchmark(args.model, args.input, args.device, args.iterations, 
                            sleep_per_image=args.sleep_per_image, verbose=args.verbose)
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
