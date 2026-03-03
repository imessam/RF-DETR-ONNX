#include "onnx_runtime.hpp"
#include <algorithm>
#include <thread>

namespace rfdetr {

OnnxRuntimeSession::OnnxRuntimeSession(const std::string &model_path,
                                       const std::string &device)
    : env_(ORT_LOGGING_LEVEL_WARNING, "RFDETRModel"),
      memory_info_(
          // Descriptor for memory location. Creates CPU RAM info with an Arena
          // allocator. Arena pre-allocates a large pool to speed up internal
          // model operations.
          Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {

  LOG_INFO("Initializing ONNX Runtime session...");
  LOG_INFO("Model path: " << model_path);
  LOG_INFO("Device preference: " << device);

  auto providers = this->getBestProviders(device);

  {
    std::stringstream ss;
    ss << "Available providers discovered: ";
    for (size_t i = 0; i < providers.size(); ++i) {
      ss << (i ? ", " : "") << providers[i];
    }
    LOG_INFO(ss.str());
  }

  bool success = false;
  std::string last_error;

  for (const auto &provider : providers) {
    try {
      LOG_INFO("Attempting to use " << provider << "...");
      Ort::SessionOptions session_options;

      // Set number of intra-op threads for parallelism
      session_options.SetIntraOpNumThreads(
          std::min(6, static_cast<int>(std::thread::hardware_concurrency())));
      session_options.SetGraphOptimizationLevel(
          GraphOptimizationLevel::ORT_ENABLE_ALL);

      if (provider == "TensorrtExecutionProvider") {
        OrtTensorRTProviderOptions trt_options{};
        session_options.AppendExecutionProvider_TensorRT(trt_options);
      } else if (provider == "CUDAExecutionProvider") {
        OrtCUDAProviderOptions cuda_options{};
        session_options.AppendExecutionProvider_CUDA(cuda_options);
      }

      this->session_ = std::make_unique<Ort::Session>(
          this->env_, model_path.c_str(), session_options);

      this->active_provider_ = provider;
      success = true;
      LOG_INFO("Successfully initialized with " << provider);
      break;
    } catch (const Ort::Exception &e) {
      last_error = e.what();
      LOG_WARN("Provider " << provider << " failed: " << last_error
                           << ". Trying next...");
    }
  }

  if (!success) {
    LOG_ERR("Failed to load ONNX model even with CPU fallback: " << last_error);
    throw std::runtime_error(
        "Failed to load ONNX model even with CPU fallback: " + last_error);
  }

  this->cacheModelMetadata();
}

namespace {
const char *tensor_type_to_string(ONNXTensorElementDataType type) {
  switch (type) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    return "float";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    return "uint8";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    return "int8";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
    return "uint16";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    return "int16";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    return "int32";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    return "int64";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
    return "string";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
    return "bool";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    return "float16";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
    return "double";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
    return "uint32";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
    return "uint64";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
    return "bfloat16";
  default:
    return "unknown";
  }
}

std::string shape_to_string(const std::vector<int64_t> &shape) {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0)
      ss << ", ";
    ss << shape[i];
  }
  ss << "]";
  return ss.str();
}
} // namespace

void OnnxRuntimeSession::cacheModelMetadata() {
  LOG_INFO("Extracting model metadata...");
  try {
    // Use an Allocator to coordinate memory ownership between C++ and the ONNX
    // library. Ort::AllocatorWithDefaultOptions ensures automatic RAII cleanup
    // for metadata strings.
    Ort::AllocatorWithDefaultOptions allocator;

    // Input metadata
    auto input_name_allocated =
        this->session_->GetInputNameAllocated(0, allocator);
    this->input_name_ = input_name_allocated.get();

    auto input_type_info = this->session_->GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    this->input_shape_ = input_tensor_info.GetShape();
    this->input_type_ = input_tensor_info.GetElementType();
    auto input_type = this->input_type_;

    LOG_INFO("Input metadata:");
    LOG_INFO("  - Name:  " << this->input_name_);
    LOG_INFO("  - Shape: " << shape_to_string(this->input_shape_));
    LOG_INFO("  - Type:  " << tensor_type_to_string(input_type));

    // Output metadata - Fix for dangling pointers
    this->num_outputs_ = this->session_->GetOutputCount();
    this->output_names_.clear();
    this->output_name_ptrs_.clear();
    this->output_names_.reserve(this->num_outputs_);
    this->output_name_ptrs_.reserve(this->num_outputs_);

    LOG_INFO("Output metadata (" << this->num_outputs_ << " outputs):");

    for (size_t i = 0; i < this->num_outputs_; ++i) {
      auto output_name_allocated =
          this->session_->GetOutputNameAllocated(i, allocator);
      std::string name = output_name_allocated.get();
      this->output_names_.emplace_back(name);

      auto output_type_info = this->session_->GetOutputTypeInfo(i);
      auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
      auto shape = output_tensor_info.GetShape();
      auto type = output_tensor_info.GetElementType();

      LOG_INFO("  Output " << i << ":");
      LOG_INFO("    - Name:  " << name);
      LOG_INFO("    - Shape: " << shape_to_string(shape));
      LOG_INFO("    - Type:  " << tensor_type_to_string(type));
    }

    // Re-build pointers from the strings we now own in this->output_names_
    for (const auto &name : this->output_names_) {
      this->output_name_ptrs_.push_back(name.c_str());
    }

    LOG_INFO("ONNX Runtime session ready.");

  } catch (const Ort::Exception &e) {
    LOG_ERR("Error during model metadata extraction: " << e.what());
    throw std::runtime_error(
        std::string("Error during model metadata extraction: ") + e.what());
  }
}

void OnnxRuntimeSession::warmup() {
  if (this->active_provider_ == "TensorrtExecutionProvider" ||
      this->active_provider_ == "CUDAExecutionProvider") {
    LOG_INFO("Warming up GPU...");

    int64_t total_size = 1;
    for (auto dim : this->input_shape_)
      total_size *= dim;

    std::vector<float> dummy_f32(total_size, 0.0f);
    std::vector<uint16_t> fp16_buf; // populated by createInputTensor if needed
    Ort::Value dummy_tensor = createInputTensor(
        dummy_f32.data(), dummy_f32.size(), this->input_shape_, fp16_buf);

    const char *input_names[] = {this->input_name_.c_str()};
    this->session_->Run(Ort::RunOptions{nullptr}, input_names, &dummy_tensor, 1,
                        this->output_name_ptrs_.data(), this->num_outputs_);

    LOG_INFO("Warmup complete.");
  }
}

void OnnxRuntimeSession::run(const cv::Mat &input_data,
                             std::vector<cv::Mat> &outputs) {
  // Thread-safety: Ort::Session::Run() is documented as thread-safe for
  // concurrent calls on the same Session object. All variables below are
  // per-call locals; the only shared state (env_, session_, memory_info_,
  // input_name_, output_names_, output_name_ptrs_) is read-only after
  // construction.

  // Only clone if not continuous (avoid redundant copy)
  const cv::Mat &input =
      input_data.isContinuous() ? input_data : input_data.clone();

  // Create ONNX tensor from OpenCV Mat
  std::vector<int64_t> input_tensor_shape = {
      1,             // batch size
      input.size[1], // channels
      input.size[2], // height
      input.size[3]  // width
  };

  // Build the input tensor, converting fp16 if the model requires it.
  // Wraps external memory (cv::Mat) as an ONNX tensor.
  // memory_info_ tells ORT that 'input.ptr' resides in CPU RAM, enabling
  // zero-copy.
  std::vector<uint16_t> fp16_buf;
  Ort::Value input_tensor = createInputTensor(input.ptr<float>(), input.total(),
                                              input_tensor_shape, fp16_buf);

  // Prepare input names
  const char *input_names[] = {this->input_name_.c_str()};

  // Run inference — output_name_ptrs_ is cached after construction; no per-call
  // alloc.
  auto output_tensors = this->session_->Run(
      Ort::RunOptions{nullptr}, input_names, &input_tensor, 1,
      this->output_name_ptrs_.data(), this->num_outputs_);

  // Convert each output tensor to a cv::Mat that owns its data.
  outputs.clear();
  outputs.reserve(this->num_outputs_);

  for (size_t i = 0; i < output_tensors.size(); ++i) {
    auto tensor_info = output_tensors[i].GetTensorTypeAndShapeInfo();
    auto shape = tensor_info.GetShape();
    float *tensor_data = output_tensors[i].GetTensorMutableData<float>();

    // ONNX shape is int64; cv::Mat dims require int
    std::vector<int> cv_shape;
    for (auto dim : shape)
      cv_shape.push_back(static_cast<int>(dim));

    // Create a view and then clone it to take ownership and ensure thread
    // safety. Cloning ensures the cv::Mat owns its buffer, allowing the
    // model's internal output_tensors to be safely destroyed.
    outputs.push_back(cv::Mat(cv_shape, CV_32F, tensor_data).clone());
  }
}

Ort::Value OnnxRuntimeSession::createInputTensor(
    const float *data_f32, size_t element_count,
    const std::vector<int64_t> &shape, std::vector<uint16_t> &fp16_buf) {

  if (input_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    // Convert float32 → float16 (IEEE 754 half-precision) using OpenCV utility.
    // cv::convertFp16 is typically SIMD-optimized and handles edge cases
    // correctly.
    cv::Mat src(1, element_count, CV_32F, const_cast<float *>(data_f32));
    cv::Mat dst;
    cv::convertFp16(src, dst);

    fp16_buf.assign(dst.ptr<uint16_t>(), dst.ptr<uint16_t>() + element_count);

    // Use the non-typed ORT overload that accepts a raw void* buffer and an
    // explicit element-type enum — the only correct way to create fp16 tensors.
    return Ort::Value::CreateTensor(
        this->memory_info_, static_cast<void *>(fp16_buf.data()),
        fp16_buf.size() * sizeof(uint16_t), shape.data(), shape.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
  }

  // Default: float32 — wrap the caller's buffer directly (zero-copy).
  return Ort::Value::CreateTensor<float>(
      this->memory_info_, const_cast<float *>(data_f32), element_count,
      shape.data(), shape.size());
}

std::vector<int64_t> OnnxRuntimeSession::getInputShape() const {
  return this->input_shape_;
}

std::string OnnxRuntimeSession::getInputName() const {
  return this->input_name_;
}

std::vector<std::string>
OnnxRuntimeSession::getBestProviders(const std::string &device) {
  auto available_providers = Ort::GetAvailableProviders();
  std::vector<std::string> providers;

  auto add_if_available = [&](const std::string &name) {
    if (std::find(available_providers.begin(), available_providers.end(),
                  name) != available_providers.end())
      providers.push_back(name);
  };

  if (device == "gpu") {
    add_if_available("TensorrtExecutionProvider");
    add_if_available("CUDAExecutionProvider");
  }

  providers.push_back("CPUExecutionProvider");
  return providers;
}

} // namespace rfdetr
