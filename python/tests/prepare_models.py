import os
import sys
import shutil
from rfdetr import RFDETRNano

def main():
    # Configuration
    checkpoint_path = "rf-detr-nano.pth"
    output_base_dir = "tests/test_models"
    final_onnx_name = "inference_model.sim.onnx"
    
    # Ensuring the output directory exists
    os.makedirs(output_base_dir, exist_ok=True)

    # 1. Initialize and download model if needed
    print(f"Initializing RFDETRNano with weights: {checkpoint_path}")
    model = RFDETRNano(pretrain_weights=checkpoint_path)
    
    # 2. Export using Roboflow native API
    # The native export creates files directly in output_dir:
    # <output_dir>/inference_model.onnx and <output_dir>/inference_model.sim.onnx
    export_dir = os.path.join(output_base_dir, "export_tmp")
    
    print(f"Exporting model to {export_dir}...")
    model.export(
        output_dir=export_dir,
        simplify=True,
        opset_version=17,
        verbose=True,
        force=True,
        batch_size=1
    )
    
    # 3. Move the simplified model to the expected test path
    exported_file = os.path.join(export_dir, "inference_model.sim.onnx")
    target_file = os.path.join(output_base_dir, final_onnx_name)
    
    if os.path.exists(exported_file):
        print(f"Moving {exported_file} to {target_file}")
        # Ensure we don't have a conflict if target already exists
        if os.path.exists(target_file):
            os.remove(target_file)
        shutil.move(exported_file, target_file)
        # Clean up temp export dir (might have inference_model.onnx too)
        if os.path.exists(export_dir):
            shutil.rmtree(export_dir)
        print("Model preparation complete.")
    else:
        print(f"Error: Exported file not found at {exported_file}")
        # List files in export_dir for debugging
        if os.path.exists(export_dir):
            print(f"Contents of {export_dir}: {os.listdir(export_dir)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
