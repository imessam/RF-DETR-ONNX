# ------------------------------------------------------------------------
# MIT License
# Copyright (c) 2026 PierreMarieCurie
# ------------------------------------------------------------------------

import argparse
import time
from src.model import RFDETRModel, DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_MAX_NUMBER_BOXES

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference with a RF-DETR ONNX model."
    )
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="Path to the ONNX model file (e.g., rf-detr-base.onnx)"
    )
    parser.add_argument(
        "--image",
        required=True,
        type=str,
        help="Path or URL to the input image"
    )
    parser.add_argument(
        "--output",
        default="output/output.jpg",
        type=str,
        help="Path to save the output image with detections (default: output/output.jpg)"
    )
    parser.add_argument(
        "--threshold",
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        type=float,
        help=f"Confidence threshold for filtering detections (default: {DEFAULT_CONFIDENCE_THRESHOLD})"
    )
    parser.add_argument(
        "--max_number_boxes",
        default=DEFAULT_MAX_NUMBER_BOXES,
        type=int,
        help=f"Maximum number of boxes to return (default: {DEFAULT_MAX_NUMBER_BOXES})"
    )
    parser.add_argument(
        "--device",
        default="gpu",
        choices=["gpu", "cpu"],
        type=str,
        help="Device to use for inference (default: gpu)"
    )
    return parser.parse_args()

def main() -> None:
    """Main inference demo function."""
    args = parse_args()

    # Initialize the model using the new RFDETRModel class
    model = RFDETRModel(args.model, device=args.device)

    # Run inference and get detections, measuring time
    _, labels, boxes, masks, timings = model.predict(args.image, args.threshold, args.max_number_boxes)
    
    # Calculate pure processing time (Pre + ORT + Post)
    processing_time = timings['preprocess'] + timings['ort_run'] + timings['postprocess']
    
    print(f"--- Inference Results ---")
    print(f"Preprocessing:  {timings['preprocess']:.2f} ms")
    print(f"ORT Run:        {timings['ort_run']:.2f} ms")
    print(f"Postprocessing: {timings['postprocess']:.2f} ms")
    print(f"---------------------------------")
    print(f"Processing (Pre+ORT+Post): {processing_time:.2f} ms")
    print(f"Processing FPS:           {1000.0 / processing_time:.2f}")
    print(f"---------------------------------")
    print(f"Total Latency (inc. I/O):  {timings['total']:.2f} ms")
    print(f"Total FPS:                {1000.0 / timings['total']:.2f}")
    print(f"---------------------------------")
    
    # Draw and save detections
    model.save_detections(args.image, boxes, labels, masks, args.output)
    print(f"Detections saved to: {args.output}")

if __name__ == "__main__":
    main()
