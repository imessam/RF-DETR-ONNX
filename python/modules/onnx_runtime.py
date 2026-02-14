# ------------------------------------------------------------------------
# MIT License
# Copyright (c) 2026 PierreMarieCurie
# ------------------------------------------------------------------------

import sys
import os 

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if base_path not in sys.path:
    sys.path.insert(0, base_path)

import onnxruntime as ort
import numpy as np

class OnnxRuntimeSession:
    """Wrapper class for ONNX Runtime session."""
    
    def __init__(self, model_path: str, device: str = "gpu"):
        """
        Initialize the ONNX Runtime session with the best available provider for the chosen device.

        Args:
            model_path (str): Path to the ONNX model file.
            device (str): Device preference ("gpu" or "cpu").
        """
        try:
            providers = self._get_best_providers(device)
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_info = self.session.get_inputs()[0]
            self.input_name = self.input_info.name
            self.input_shape = self.input_info.shape

            print(f"Input shape: {self.input_shape}")
            print(f"providers: {providers}")
            
            active_providers = self.session.get_providers()
            print(f"active_providers: {active_providers}")
            print(f"--- ONNX Runtime: Using {active_providers[0]} for inference ---")
            
            # Perform a warmup run to initialize CUDA/TensorRT
            if "TensorrtExecutionProvider" in active_providers or "CUDAExecutionProvider" in active_providers:
                print("--- ONNX Runtime: Warming up GPU... ---")
                dummy_input = np.zeros(self.input_shape, dtype=np.float32)
                self.session.run(None, {self.input_name: dummy_input})
                print("--- ONNX Runtime: Warmup complete ---")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load ONNX model from '{model_path}'. "
                f"Ensure the path is correct and the model is a valid ONNX file."
            ) from e

    def _get_best_providers(self, device: str = "gpu") -> list[str]:
        """
        Determine the best available execution providers based on device preference.
        
        Args:
            device (str): "gpu" (TensorRT > CUDA > CPU) or "cpu" (CPU only).
        """
        available = ort.get_available_providers()
        providers = []
        
        if device.lower() == "gpu":
            if "TensorrtExecutionProvider" in available:
                providers.append("TensorrtExecutionProvider")
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
        
        providers.append("CPUExecutionProvider")
        return providers

    def run(self, input_data: np.ndarray) -> list[np.ndarray]:
        """Run inference with the provided input data."""
        return self.session.run(None, {self.input_name: input_data})

    def get_input_shape(self) -> list[int]:
        """Get the expected input shape of the model."""
        return self.input_shape

    def get_input_name(self) -> str:
        """Get the name of the input tensor."""
        return self.input_name

    def get_inputs(self) -> list[ort.NodeArg]:
        """Get all input information."""
        return self.session.get_inputs()
