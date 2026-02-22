#include "onnx_runtime.hpp"
#include <algorithm>
#include <iostream>

namespace rfdetr {

OnnxRuntimeSession::OnnxRuntimeSession(const std::string &modelPath,
                                       const std::string &device)
    : env_(ORT_LOGGING_LEVEL_WARNING, "RFDETRModel"),
      memoryInfo_(
          Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {

  auto providers = getBestProviders(device);
  bool success = false;
  std::string lastError;

  // Try providers in priority order with fallback
  for (size_t i = 0; i < providers.size(); ++i) {
    try {
      Ort::SessionOptions sessionOptions;
      sessionOptions.SetGraphOptimizationLevel(
          GraphOptimizationLevel::ORT_ENABLE_ALL);

      // Add providers from current priority down to CPU
      for (size_t j = i; j < providers.size(); ++j) {
        if (providers[j] == "TensorrtExecutionProvider") {
          std::cout << "Attempting to use TensorRT provider..." << std::endl;
          OrtTensorRTProviderOptions tensorrtOptions{};
          sessionOptions.AppendExecutionProvider_TensorRT(tensorrtOptions);
        } else if (providers[j] == "CUDAExecutionProvider") {
          std::cout << "Attempting to use CUDA provider..." << std::endl;
          OrtCUDAProviderOptions cudaOptions{};
          sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
        }
      }

      // Create session
      session_ = std::make_unique<Ort::Session>(env_, modelPath.c_str(),
                                                sessionOptions);

      activeProvider_ = providers[i];
      success = true;
      break;
    } catch (const Ort::Exception &e) {
      lastError = e.what();
      std::cerr << "Warning: Failed to initialize session with provider "
                << providers[i] << ": " << lastError << std::endl;
      std::cerr << "Falling back to next available provider..." << std::endl;
      continue;
    }
  }

  if (!success) {
    throw std::runtime_error(
        "Failed to load ONNX model even with CPU fallback: " + lastError);
  }

  try {
    // Get input information
    Ort::AllocatorWithDefaultOptions allocator;
    auto inputNameAllocated = session_->GetInputNameAllocated(0, allocator);
    inputName_ = inputNameAllocated.get();

    auto inputTypeInfo = session_->GetInputTypeInfo(0);
    auto tensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    inputShape_ = tensorInfo.GetShape();

    // Get output information and cache output names
    numOutputs_ = session_->GetOutputCount();
    cachedOutputNamesPtr_.clear();
    for (size_t i = 0; i < numOutputs_ && i < 3; ++i) {
      auto outputNameAllocated = session_->GetOutputNameAllocated(i, allocator);
      outputNames_[i] = outputNameAllocated.get();
    }
    // Build the cached pointer vector once
    for (size_t i = 0; i < numOutputs_; ++i) {
      cachedOutputNamesPtr_.push_back(outputNames_[i].c_str());
    }

    std::cout << "Input shape: [";
    for (size_t i = 0; i < inputShape_.size(); ++i) {
      std::cout << inputShape_[i];
      if (i < inputShape_.size() - 1)
        std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "--- ONNX Runtime: Session created successfully ---"
              << std::endl;

    std::cout << "--- ONNX Runtime: Using " << activeProvider_
              << " for inference ---" << std::endl;

  } catch (const Ort::Exception &e) {
    throw std::runtime_error(
        std::string("Error during model metadata extraction: ") + e.what());
  }
}

void OnnxRuntimeSession::warmup() {
  if (activeProvider_ == "TensorrtExecutionProvider" ||
      activeProvider_ == "CUDAExecutionProvider") {
    std::cout << "--- ONNX Runtime: Warming up GPU... ---" << std::endl;

    // Create dummy input matching the expected shape
    int64_t totalSize = 1;
    for (auto dim : inputShape_) {
      totalSize *= dim;
    }

    std::vector<float> dummyData(totalSize, 0.0f);
    Ort::Value dummyTensor = Ort::Value::CreateTensor<float>(
        memoryInfo_, dummyData.data(), dummyData.size(), inputShape_.data(),
        inputShape_.size());

    const char *inputNames[] = {inputName_.c_str()};

    session_->Run(Ort::RunOptions{nullptr}, inputNames, &dummyTensor, 1,
                  cachedOutputNamesPtr_.data(), numOutputs_);

    std::cout << "--- ONNX Runtime: Warmup complete ---" << std::endl;
  }
}

void OnnxRuntimeSession::run(const cv::Mat &inputData,
                             std::vector<cv::Mat> &outputs) {
  // Only clone if not continuous (avoid redundant copy)
  const cv::Mat &input =
      inputData.isContinuous() ? inputData : inputData.clone();

  // Create ONNX tensor from OpenCV Mat
  std::vector<int64_t> inputTensorShape = {
      1,
      input.size[1], // channels
      input.size[2], // height
      input.size[3]  // width
  };

  Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
      memoryInfo_, const_cast<float *>(input.ptr<float>()), input.total(),
      inputTensorShape.data(), inputTensorShape.size());

  // Prepare input names
  const char *inputNames[] = {inputName_.c_str()};

  // Run inference
  lastOutputTensors_ =
      session_->Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1,
                    cachedOutputNamesPtr_.data(), numOutputs_);

  // Convert output tensors to OpenCV Mats (Zero-copy)
  outputs.clear();
  outputs.reserve(numOutputs_);

  for (size_t i = 0; i < lastOutputTensors_.size(); ++i) {
    auto tensorInfo = lastOutputTensors_[i].GetTensorTypeAndShapeInfo();
    auto shape = tensorInfo.GetShape();

    // Get pointer to tensor data
    float *tensorData = lastOutputTensors_[i].GetTensorMutableData<float>();

    // Convert shape to OpenCV dims
    std::vector<int> cvShape;
    for (auto dim : shape) {
      cvShape.push_back(static_cast<int>(dim));
    }

    // Create Mat wrapping the pointer (Zero-copy)
    outputs.emplace_back(cvShape, CV_32F, tensorData);
  }
}

std::vector<int64_t> OnnxRuntimeSession::getInputShape() const {
  return inputShape_;
}

std::string OnnxRuntimeSession::getInputName() const { return inputName_; }

std::vector<std::string>
OnnxRuntimeSession::getBestProviders(const std::string &device) {
  auto availableProviders = Ort::GetAvailableProviders();
  std::vector<std::string> providers;

  if (device == "gpu") {
    // Check for TensorRT
    if (std::find(availableProviders.begin(), availableProviders.end(),
                  "TensorrtExecutionProvider") != availableProviders.end()) {
      providers.push_back("TensorrtExecutionProvider");
    }

    // Check for CUDA
    if (std::find(availableProviders.begin(), availableProviders.end(),
                  "CUDAExecutionProvider") != availableProviders.end()) {
      providers.push_back("CUDAExecutionProvider");
    }
  }

  // Always add CPU as fallback
  providers.push_back("CPUExecutionProvider");

  return providers;
}

} // namespace rfdetr
