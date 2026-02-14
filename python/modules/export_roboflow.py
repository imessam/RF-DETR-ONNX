import argparse
import sys
import os 

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if base_path not in sys.path:
    sys.path.insert(0, base_path)

from rfdetr import RFDETRNano, RFDETRSmall, RFDETRBase, RFDETRMedium, RFDETRLarge

MODELS = {
    "nano": RFDETRNano,
    "small": RFDETRSmall,
    "base": RFDETRBase,
    "medium": RFDETRMedium,
    "large": RFDETRLarge
}

def parse_args():
    parser = argparse.ArgumentParser(description="Export RF-DETR model via Roboflow API")
    parser.add_argument("--weights", type=str, required=True, help="Path to the checkpoint (.pth) file")
    parser.add_argument("--model-type", type=str, default="nano", choices=MODELS.keys(), help="Model architecture type")
    parser.add_argument("--output-dir", type=str, default=os.path.join(base_path, "models"), help="Directory to save the exported model")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for export")
    parser.add_argument("--no-simplify", action="store_true", help="Disable ONNX simplification")
    return parser.parse_args()

def main():
    args = parse_args()
    
    model_class = MODELS[args.model_type]
    print(f"Initializing {model_class.__name__} with weights: {args.weights}")
    model = model_class(pretrain_weights=args.weights)

    file_name = os.path.basename(args.weights).split(".")[0]
    output_dir = os.path.join(args.output_dir, file_name)

    print(f"Exporting to {output_dir} (opset {args.opset})...")
    model.export(
        output_dir=output_dir,
        simplify=not args.no_simplify,
        opset_version=args.opset,
        verbose=True,
        force=True,
        shape=None,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()