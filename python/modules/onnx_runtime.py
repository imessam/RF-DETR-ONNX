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
            sess_options = ort.SessionOptions()
            # ORT_ENABLE_ALL allows the optimizer to convert FP16 nodes to FP32
            # when running on CPU (which does not natively support float16 tensors).
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(
                model_path, sess_options=sess_options, providers=providers
            )
            self.input_info = self.session.get_inputs()[0]
            self.input_name = self.input_info.name
            self.input_shape = self.input_info.shape
            self.input_dtype = self._ort_type_to_numpy_dtype(self.input_info.type)

            print("Input metadata:")
            print(f"  - Name:  {self.input_name}")
            print(f"  - Shape: {self.input_shape}")
            print(f"  - Type:  {self.input_info.type} -> numpy: {self.input_dtype}")
            
            active_providers = self.session.get_providers()
            print(f"--- ONNX Runtime: Using {active_providers[0]} for inference ---")
            
            outputs = self.session.get_outputs()
            print(f"Output metadata ({len(outputs)} outputs):")
            for i, output in enumerate(outputs):
                print(f"  Output {i}:")
                print(f"    - Name:  {output.name}")
                print(f"    - Shape: {output.shape}")
                print(f"    - Type:  {output.type}")

            # Perform a warmup run to initialize CUDA/TensorRT
            if "TensorrtExecutionProvider" in active_providers or "CUDAExecutionProvider" in active_providers:
                print("--- ONNX Runtime: Warming up GPU... ---")
                dummy_input = np.zeros(self.input_shape, dtype=self.input_dtype)
                self.session.run(None, {self.input_name: dummy_input})
                print("--- ONNX Runtime: Warmup complete ---")
        except Exception as e:
            print(f"ERROR: Failed to load model '{model_path}' on {device}:")
            print(f"  {str(e)}")
            raise

    @staticmethod
    def _ort_type_to_numpy_dtype(ort_type: str) -> np.dtype:
        """
        Map an ONNX Runtime type string (e.g. 'tensor(float16)') to a numpy dtype.

        Args:
            ort_type (str): The type string returned by NodeArg.type.

        Returns:
            np.dtype: Corresponding numpy dtype (defaults to float32 for unknown types).
        """
        mapping = {
            "tensor(float16)": np.float16,
            "tensor(float)": np.float32,
            "tensor(double)": np.float64,
            "tensor(int8)": np.int8,
            "tensor(int16)": np.int16,
            "tensor(int32)": np.int32,
            "tensor(int64)": np.int64,
            "tensor(uint8)": np.uint8,
        }
        return np.dtype(mapping.get(ort_type, np.float32))

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
        """Run inference with the provided input data, casting to the model's expected dtype."""
        if input_data.dtype != self.input_dtype:
            input_data = input_data.astype(self.input_dtype)
        return self.session.run(None, {self.input_name: input_data})

    def get_input_shape(self) -> list[int]:
        """Get the expected input shape of the model."""
        return self.input_shape

    def get_input_name(self) -> str:
        """Get the name of the input tensor."""
        return self.input_name

    def get_inputs(self) -> list:
        """Get all input information."""
        return self.session.get_inputs()
