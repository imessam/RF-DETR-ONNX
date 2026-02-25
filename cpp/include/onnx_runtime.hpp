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

  Ort::Env env_;
  std::unique_ptr<Ort::Session> session_;
  Ort::MemoryInfo memoryInfo_;
  std::vector<int64_t> inputShape_;
  std::string inputName_;
  std::string activeProvider_;
  std::vector<std::string> outputNames_; // Output names: boxes, scores, [masks]
  size_t numOutputs_;
};

} // namespace rfdetr
