# ------------------------------------------------------------------------
# MIT License
# Copyright (c) 2026 PierreMarieCurie
#
# Portions of this file use RF-DETR
# Copyright (c) Roboflow
# Licensed under the Apache License, Version 2.0
# See https://www.apache.org/licenses/LICENSE-2.0 for details.
# ------------------------------------------------------------------------

import argparse
import torch
from pathlib import Path
from typing import Optional
import warnings
from rfdetr.detr import (
    RFDETRNano,
    RFDETRSmall,
    RFDETRBase,
    RFDETRMedium, 
    RFDETRLargeNew,
    RFDETRLargeDeprecated,
    RFDETRSegPreview,
    RFDETRSegNano,
    RFDETRSegSmall,
    RFDETRSegMedium, 
    RFDETRSegLarge,
    RFDETRSegXLarge,
    RFDETRSeg2XLarge
)
from rfdetr.platform.models import (
    RFDETR2XLarge,
    RFDETRXLarge,
)
import onnx
from onnxsim import simplify

# Backbone params to model class mapping
BACKBONE_LEN_TO_CLASS = {
    23_266_048: RFDETRNano,  # rf-detr-nano.pth
    23_438_080: RFDETRSmall,  # rf-detr-small.pth
    23_501_440: RFDETRBase,  # rf-detr-base.pth, rf-detr-base-o365.pth
    23_542_528: RFDETRMedium,  # rf-detr-medium.pth
    23_788_288: RFDETRLargeNew,  # rf-detr-large-2026.pth
    92_713_216: RFDETRXLarge,  # rf-detr-xlarge.pth 
    93_259_264: RFDETR2XLarge,  # rf-detr-xxlarge.pth
    119_037_696: RFDETRLargeDeprecated,  # rf-detr-large.pth
    23_175_424: RFDETRSegNano,  # rf-detr-seg-nano.pt
    23_309_056: RFDETRSegSmall,  # rf-detr-seg-small.pt
    23_593_216: RFDETRSegLarge,  # rf-detr-seg-large.pt
    23_954_176: RFDETRSegXLarge,  # rf-detr-seg-xlarge.pt
    24_488_704: RFDETRSeg2XLarge,  # rf-detr-seg-xxlarge.pt
}

def load_rfdetr_model(cpkt: str):
    try:
        # Load checkpoint
        print(f"Loading checkpoint: {cpkt}...")
        obj = torch.load(cpkt , weights_only=False)
        args = obj.get("args", None)
        
        # Guess model class based on backbone size
        backbone_params = sum(p.numel() for k, p in obj["model"].items() if "backbone" in k and isinstance(p, torch.Tensor))
        model_class = None
        if backbone_params == 23_413_504:
            if 'transformer.decoder.layers.4.self_attn.in_proj_weight' in obj["model"].keys():
                model_class = RFDETRSegMedium
            else:
                model_class = RFDETRSegPreview
        else:
            model_class = BACKBONE_LEN_TO_CLASS.get(backbone_params, None)
        
        # Return model instance
        if model_class is None:
            raise ValueError(f"Unknown model architecture")
        print(f"Assuming the checkpoint corresponds to {model_class.__name__}")
        if issubclass(model_class, (RFDETRXLarge, RFDETR2XLarge)):
            warnings.warn(
                f"\n{'='*80}\n"
                f"WARNING: The model '{model_class.__name__}' requires accepting the platform model license.\n"
                f"License URL: https://roboflow.com/platform-model-license-1-0\n"
                f"These models require a commercial license to use in commercial applications.\n"
                f"By loading this model, you are agreeing to the terms of the license.\n"
                f"{'='*80}",
                UserWarning,
                stacklevel=2
            )
        return model_class(pretrain_weights=cpkt, accept_platform_model_license=True)
        
    except Exception as e:
        raise RuntimeError(f"Error loading RF-DETR model from checkpoint '{cpkt}'") from e

def export_onnx(checkpoint: str, onnx_name: Optional[str] = None, no_simplify: bool = False) -> None:
    """
    Export an RF-DETR model to ONNX format.

    Args:
        checkpoint (str): Path to the model checkpoint file (e.g. "rf-detr-nano.pth").
        onnx_name (str): Name of the output ONNX file.
    """
    try:
        # Validate inputs
        if not isinstance(checkpoint, str):
            raise TypeError(f"Expected 'checkpoint' to be a str, got {type(checkpoint).__name__}")
        path = Path(checkpoint)
        if path.suffix not in {".pt", ".pth"}:
            raise ValueError(f"Invalid model file: {checkpoint}. Expected a .pt or .pth file.")
        if onnx_name is None:
            onnx_name = path.with_suffix(".onnx").name

        # Load RF-DETR model
        rfdetr_model = load_rfdetr_model(checkpoint)
        model = rfdetr_model.model.model
        config = rfdetr_model.model_config

        # Forward pass to get output layer names
        resolution = config.resolution
        device = config.device
        dummy_input = torch.randn(1, 3, resolution, resolution, device=device)
        model.eval()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.no_grad():
                output = model(dummy_input)
        if len(output) == 4: # Object detection
            output_names = list(output.keys())[:2][::-1]
        elif len(output) == 5: # Instance segmentation
            output_names = list(output.keys())[:3][::-1]
        else:
            raise ValueError("Unexpected model output structure")

        # Export to ONNX
        print(f"Exporting model to ONNX: {onnx_name}...")
        model.export()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(
                model,
                dummy_input,
                onnx_name,
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=["input"],
                output_names=output_names
            )
        
        # Simplify ONNX model
        if not no_simplify:
            print("Simplifying ONNX model...")
            onnx_model = onnx.load(onnx_name)
            model_simplified, check = simplify(onnx_model)
            if check:
                onnx.save(model_simplified, onnx_name)
            else:
                raise RuntimeError("ONNX simplification check failed")
        
        print(f"Model successfully exported to {onnx_name}")

    except Exception as e:
        raise RuntimeError(f"Failed to export model from checkpoint '{checkpoint}'") from e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export RF-DETR model to ONNX format"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a custom RF-DETR checkpoint (.pth for object detection, .pt for instance segmentation)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Name of the output ONNX model file. If not provided, checkpoint name will be used with .onnx extension",
    )
    parser.add_argument(
    "--no-simplify",
    action="store_true",
    help="Do not simplify the ONNX model (diseabled by default)",
    )
    args = parser.parse_args()

    export_onnx(args.checkpoint, args.model_name, args.no_simplify)