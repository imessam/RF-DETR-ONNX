

"""
Export an ONNX model to FP16 (float16) or mixed-precision.

References:
    https://onnxruntime.ai/docs/performance/model-optimizations/float16.html

Requirements:
    pip install onnx onnxconverter-common
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

try:
    import onnx
except ImportError:
    print("Error: 'onnx' is not installed. Run: pip install onnx", file=sys.stderr)
    sys.exit(1)

try:
    from onnxconverter_common import float16
    from onnxconverter_common import auto_mixed_precision
except ImportError:
    print(
        "Error: 'onnxconverter-common' is not installed. Run: pip install onnxconverter-common",
        file=sys.stderr,
    )
    sys.exit(1)


def convert_to_fp16(
    input_path: str,
    output_path: Optional[str] = None,
    keep_io_types: bool = False,
    min_positive_val: float = 1e-7,
    max_finite_val: float = 1e4,
    disable_shape_infer: bool = False,
    op_block_list: Optional[list] = None,
    node_block_list: Optional[list] = None,
) -> None:
    """
    Convert an ONNX model from float32 to float16.

    Args:
        input_path (str): Path to the input ONNX model.
        output_path (str, optional): Path for the output FP16 ONNX model.
            Defaults to ``<input_stem>_fp16.onnx`` in the same directory.
        keep_io_types (bool): Keep model inputs/outputs as float32. Default: False.
        min_positive_val (float): Minimum positive constant value after clipping.
            0.0, nan, inf, and -inf are always unchanged. Default: 1e-7.
        max_finite_val (float): Maximum finite constant value after clipping.
            Default: 1e4.
        disable_shape_infer (bool): Skip ONNX shape/type inference. Useful if
            inference crashes or shapes/types are already present. Default: False.
        op_block_list (list, optional): Op types to keep in float32. Defaults to
            ``float16.DEFAULT_OP_BLOCK_LIST`` (ops unsupported in FP16 by ORT).
        node_block_list (list, optional): Node names to keep in float32.
    """
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input ONNX model not found: {input_path}")
    if input_path.suffix != ".onnx":
        raise ValueError(f"Expected an .onnx file, got: {input_path}")

    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_fp16.onnx"
    else:
        output_path = Path(output_path)

    print(f"Loading model: {input_path}")
    model = onnx.load(str(input_path))

    print("Converting to FP16...")
    model_fp16 = float16.convert_float_to_float16(
        model,
        min_positive_val=min_positive_val,
        max_finite_val=max_finite_val,
        keep_io_types=keep_io_types,
        disable_shape_infer=disable_shape_infer,
        op_block_list=op_block_list,
        node_block_list=node_block_list,
    )

    onnx.save(model_fp16, str(output_path))
    print(f"FP16 model saved to: {output_path}")


def convert_to_mixed_precision(
    input_path: str,
    feed_dict: dict,
    output_path: Optional[str] = None,
    rtol: float = 0.01,
    atol: float = 0.001,
    keep_io_types: bool = True,
) -> None:
    """
    Convert an ONNX model to mixed-precision (FP16 where possible, FP32 where needed).

    This tool finds the minimal set of ops to keep in float32 while retaining
    accuracy. A GPU is required because the CPU ORT does not support FP16 ops.

    Args:
        input_path (str): Path to the input ONNX model.
        feed_dict (dict): Sample inputs as ``{input_name: np.ndarray}``.
        output_path (str, optional): Path for the output mixed-precision ONNX model.
            Defaults to ``<input_stem>_mixed_fp16.onnx`` in the same directory.
        rtol (float): Relative tolerance for accuracy comparison. Default: 0.01.
        atol (float): Absolute tolerance for accuracy comparison. Default: 0.001.
        keep_io_types (bool): Keep model input/outputs as float32. Default: True.
    """
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input ONNX model not found: {input_path}")
    if input_path.suffix != ".onnx":
        raise ValueError(f"Expected an .onnx file, got: {input_path}")

    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_mixed_fp16.onnx"
    else:
        output_path = Path(output_path)

    print(f"Loading model: {input_path}")
    model = onnx.load(str(input_path))

    print("Converting to mixed precision (requires GPU)...")
    model_fp16 = auto_mixed_precision.auto_convert_mixed_precision(
        model,
        feed_dict,
        rtol=rtol,
        atol=atol,
        keep_io_types=keep_io_types,
    )

    onnx.save(model_fp16, str(output_path))
    print(f"Mixed-precision model saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert an ONNX model to FP16 or mixed-precision FP16.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input ONNX model (.onnx).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Path for the output ONNX model. "
            "Defaults to '<input_stem>_fp16.onnx' (or '_mixed_fp16.onnx' for mixed mode)."
        ),
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help=(
            "Use auto mixed-precision conversion instead of full FP16. "
            "Requires a GPU and a sample input (see --sample-input). "
            "Finds the minimal set of ops to keep in float32 to retain accuracy."
        ),
    )
    parser.add_argument(
        "--keep-io-types",
        action="store_true",
        help="Keep model inputs and outputs as float32 (recommended for compatibility).",
    )

    # FP16-specific arguments
    fp16_group = parser.add_argument_group("FP16 conversion options (ignored in mixed-precision mode)")
    fp16_group.add_argument(
        "--min-positive-val",
        type=float,
        default=1e-7,
        help="Minimum positive constant value after FP16 clipping.",
    )
    fp16_group.add_argument(
        "--max-finite-val",
        type=float,
        default=1e4,
        help="Maximum finite constant value after FP16 clipping.",
    )
    fp16_group.add_argument(
        "--disable-shape-infer",
        action="store_true",
        help="Skip ONNX shape/type inference during conversion.",
    )
    fp16_group.add_argument(
        "--op-block-list",
        type=str,
        nargs="+",
        default=None,
        metavar="OP_TYPE",
        help=(
            "Op types to leave in float32 (e.g. 'Softmax' 'LayerNormalization'). "
            "By default uses float16.DEFAULT_OP_BLOCK_LIST."
        ),
    )
    fp16_group.add_argument(
        "--node-block-list",
        type=str,
        nargs="+",
        default=None,
        metavar="NODE_NAME",
        help="Node names to leave in float32.",
    )

    # Mixed-precision specific arguments
    mp_group = parser.add_argument_group("Mixed-precision options (only used with --mixed-precision)")
    mp_group.add_argument(
        "--sample-input",
        type=str,
        default=None,
        metavar="NPY_FILE",
        help=(
            "Path to a .npy file containing a sample input array for the model. "
            "Required when --mixed-precision is set. "
            "The input name is taken from the model's first input."
        ),
    )
    mp_group.add_argument(
        "--rtol",
        type=float,
        default=0.01,
        help="Relative tolerance for mixed-precision accuracy comparison.",
    )
    mp_group.add_argument(
        "--atol",
        type=float,
        default=0.001,
        help="Absolute tolerance for mixed-precision accuracy comparison.",
    )

    args = parser.parse_args()

    if args.mixed_precision:
        if args.sample_input is None:
            parser.error("--sample-input is required when using --mixed-precision.")

        import numpy as np

        sample = np.load(args.sample_input)
        # Use the model's first input name as the key
        _model_tmp = onnx.load(args.input)
        input_name = _model_tmp.graph.input[0].name

        feed_dict = {input_name: sample}
        convert_to_mixed_precision(
            input_path=args.input,
            feed_dict=feed_dict,
            output_path=args.output,
            rtol=args.rtol,
            atol=args.atol,
            keep_io_types=args.keep_io_types,
        )
    else:
        convert_to_fp16(
            input_path=args.input,
            output_path=args.output,
            keep_io_types=args.keep_io_types,
            min_positive_val=args.min_positive_val,
            max_finite_val=args.max_finite_val,
            disable_shape_infer=args.disable_shape_infer,
            op_block_list=args.op_block_list,
            node_block_list=args.node_block_list,
        )
