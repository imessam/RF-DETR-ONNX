#pragma once

#include <memory>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace rfdetr {

/**
 * @brief Wrapper class for ONNX Runtime session management
 *
 * Handles model loading, execution provider selection, and inference execution.
 * Uses RAII for automatic resource management.
 */
class OnnxRuntimeSession {
public:
  /**
   * @brief Initialize ONNX Runtime session with the best available provider
   * @param modelPath Path to the ONNX model file
   * @param device Device preference ("gpu" or "cpu")
   * @throws std::runtime_error if model loading fails
   */
  OnnxRuntimeSession(const std::string &modelPath,
                     const std::string &device = "gpu");

  /**
   * @brief Run inference on input data
   * @param inputData Input image tensor (1, C, H, W)
   * @param outputs Output tensors from the model
   */
  void run(const cv::Mat &inputData, std::vector<cv::Mat> &outputs);

  /**
   * @brief Get the expected input shape of the model
   * @return Vector containing input dimensions [batch, channels, height, width]
   */
  std::vector<int64_t> getInputShape() const;

  /**
   * @brief Perform a warmup run to initialize GPU/TensorRT resources
   *
   * Highly recommended for GPU providers to avoid latency spikes on the first
   * inference.
   */
  void warmup();

  /**
   * @brief Get the name of the input tensor
   * @return Input tensor name
   */
  std::string getInputName() const;

private:
  /**
   * @brief Determine the best available execution providers
   * @param device Device preference ("gpu" or "cpu")
   * @return List of provider names in priority order
   */
  std::vector<std::string> getBestProviders(const std::string &device);

  /**
   * @brief Read and cache model input/output metadata from the session.
   * @throws std::runtime_error on ONNX metadata extraction failure.
   */
  void cacheModelMetadata();

  Ort::Env env_;
  std::unique_ptr<Ort::Session> session_;

  Ort::MemoryInfo memory_info_;
  std::string active_provider_;

  std::vector<int64_t> input_shape_;
  std::string input_name_;
  ONNXTensorElementDataType input_type_{ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT};

  // output_names_ owns the string data; output_name_ptrs_ holds non-owning
  // raw pointers into those strings for zero-alloc passing to Session::Run().
  std::vector<std::string> output_names_;
  std::vector<const char *> output_name_ptrs_;
  size_t num_outputs_;

  /**
   * @brief Create an input Ort::Value with the correct element type.
   *
   * Dispatches to CreateTensor<float> or CreateTensor<uint16_t> (fp16)
   * depending on what the model declared.
   *
   * @param data_f32  Source data always provided as float32.
   *                  For fp16 models the data is converted internally.
   * @param shape     Tensor shape.
   * @param fp16_buf  Buffer that owns the converted fp16 data (when needed).
   * @return Ort::Value ready for Session::Run().
   */
  Ort::Value createInputTensor(const float *data_f32, size_t element_count,
                               const std::vector<int64_t> &shape,
                               std::vector<uint16_t> &fp16_buf);
};

} // namespace rfdetr
