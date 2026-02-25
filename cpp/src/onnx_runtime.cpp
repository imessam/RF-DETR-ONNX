#include "onnx_runtime.hpp"
#include <algorithm>
#include <iostream>

namespace rfdetr {

OnnxRuntimeSession::OnnxRuntimeSession(const std::string &modelPath,
                                       const std::string &device)
    : env_(ORT_LOGGING_LEVEL_WARNING, "RFDETRModel"),
      memoryInfo_(
          // Descriptor for memory location. Creates CPU RAM info with an Arena
          // allocator. Arena pre-allocates a large pool to speed up internal
          // model operations.
          Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {

  auto providers = getBestProviders(device);
  bool success = false;
  std::string lastError;

  // Try each provider in priority order, falling back to the next on failure.
  // CPU is always last in the list so at least one provider will succeed.
  for (const auto &provider : providers) {
    try {
      Ort::SessionOptions sessionOptions;
      sessionOptions.SetGraphOptimizationLevel(
          GraphOptimizationLevel::ORT_ENABLE_ALL);

      if (provider == "TensorrtExecutionProvider") {
        std::cout << "Attempting to use TensorRT provider..." << std::endl;
        OrtTensorRTProviderOptions tensorrtOptions{};
        sessionOptions.AppendExecutionProvider_TensorRT(tensorrtOptions);
      } else if (provider == "CUDAExecutionProvider") {
        std::cout << "Attempting to use CUDA provider..." << std::endl;
        OrtCUDAProviderOptions cudaOptions{};
        sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
      }
      // CPUExecutionProvider is the ORT default — no explicit append needed.

      session_ = std::make_unique<Ort::Session>(env_, modelPath.c_str(),
                                                sessionOptions);
      activeProvider_ = provider;
      success = true;
      break;
    } catch (const Ort::Exception &e) {
      lastError = e.what();
      std::cerr << "Warning: provider " << provider << " failed: " << lastError
                << ". Trying next..." << std::endl;
    }
  }

  if (!success) {
    throw std::runtime_error(
        "Failed to load ONNX model even with CPU fallback: " + lastError);
  }

  try {
    // Use an Allocator to coordinate memory ownership between C++ and the ONNX
    // library. Ort::AllocatorWithDefaultOptions ensures automatic RAII cleanup
    // for metadata strings.
    Ort::AllocatorWithDefaultOptions allocator;
    auto inputNameAllocated = session_->GetInputNameAllocated(0, allocator);
    inputName_ = inputNameAllocated.get();

    auto inputTypeInfo = session_->GetInputTypeInfo(0);
    auto tensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    inputShape_ = tensorInfo.GetShape();

    // Get output names
    numOutputs_ = session_->GetOutputCount();
    outputNames_.resize(numOutputs_);
    for (size_t i = 0; i < numOutputs_; ++i) {
      auto outputNameAllocated = session_->GetOutputNameAllocated(i, allocator);
      outputNames_[i] = outputNameAllocated.get();
    }

    std::cout << "Input shape: [";
    for (size_t i = 0; i < inputShape_.size(); ++i)
      std::cout << (i ? ", " : "") << inputShape_[i];
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

    std::vector<const char *> inputNames = {inputName_.c_str()};
    std::vector<const char *> outputNames;
    for (const auto &name : outputNames_) {
      outputNames.push_back(name.c_str());
    }

    session_->Run(Ort::RunOptions{nullptr}, inputNames.data(), &dummyTensor, 1,
                  outputNames.data(), numOutputs_);

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
      1,             // batch size
      input.size[1], // channels
      input.size[2], // height
      input.size[3]  // width
  };

  // Wraps external memory (cv::Mat) as an ONNX tensor.
  // memoryInfo_ tells ORT that 'input.ptr' resides in CPU RAM, enabling
  // zero-copy.
  Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
      memoryInfo_, const_cast<float *>(input.ptr<float>()), input.total(),
      inputTensorShape.data(), inputTensorShape.size());

  // Prepare input names
  const char *inputNames[] = {inputName_.c_str()};

  // Prepare output names
  std::vector<const char *> outputNames;
  for (const auto &name : outputNames_) {
    outputNames.push_back(name.c_str());
  }

  // Run inference
  auto outputTensors =
      session_->Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1,
                    outputNames.data(), numOutputs_);

  // Convert each output tensor to a cv::Mat that owns its data.
  outputs.clear();
  outputs.reserve(numOutputs_);

  for (size_t i = 0; i < outputTensors.size(); ++i) {
    auto tensorInfo = outputTensors[i].GetTensorTypeAndShapeInfo();
    auto shape = tensorInfo.GetShape();

    float *tensorData = outputTensors[i].GetTensorMutableData<float>();

    // ONNX shape is int64; cv::Mat dims require int
    std::vector<int> cvShape;
    for (auto dim : shape)
      cvShape.push_back(static_cast<int>(dim));

    // Create a view and then clone it to take ownership and ensure thread
    // safety. Cloning ensures the cv::Mat owns its buffer, allowing the model's
    // internal outputTensors to be safely destroyed.
    outputs.push_back(cv::Mat(cvShape, CV_32F, tensorData).clone());
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
