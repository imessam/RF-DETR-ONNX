#include "rfdetr_model.hpp"
#include "logger.hpp"
#include "utils.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

namespace rfdetr {

RFDETRModel::RFDETRModel(const std::string &model_path,
                         const std::string &device) {
  LOG_INFO("Initializing RF-DETR model...");

  this->ort_session_ = std::make_unique<OnnxRuntimeSession>(model_path, device);

  auto input_shape = this->ort_session_->getInputShape();
  this->input_height_ = static_cast<int>(input_shape[2]);
  this->input_width_ = static_cast<int>(input_shape[3]);

  LOG_INFO("Model resolution: " << this->input_width_ << "x"
                                << this->input_height_);

  this->warmup();
}

void RFDETRModel::warmup() { this->ort_session_->warmup(); }

void RFDETRModel::preprocess(const cv::Mat &image, cv::Mat &output) {
  cv::Mat rgb;
  cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);

  cv::Mat resized;
  cv::resize(rgb, resized, cv::Size(this->input_width_, this->input_height_));

  const int plane_size = this->input_height_ * this->input_width_;

  cv::Mat float_rgb;
  resized.convertTo(float_rgb, CV_32FC3, 1.0 / 255.0);

  cv::Mat channels[3];
  cv::split(float_rgb, channels);

  // Normalize using ImageNet statistics (Mean/StdDev) to match training
  // distribution. Then transform memory layout from HWC (OpenCV style) to CHW
  // (ONNX style). memcpy copies each color plane (R, G, B) into a single,
  // contiguous memory row.
  //
  // Thread-safety: thread_local gives each thread its own buffer so concurrent
  // calls to predict() never race. resize() is a no-op after the first call on
  // a given thread (same model dimensions), so there is no per-frame alloc.
  thread_local std::vector<float> local_buffer;
  local_buffer.resize(3 * plane_size);
  float *buf_ptr = local_buffer.data();
  for (int c = 0; c < 3; ++c) {
    channels[c] = (channels[c] - this->MEANS[c]) / this->STDS[c];
    std::memcpy(buf_ptr + c * plane_size, channels[c].data,
                plane_size * sizeof(float));
  }

  // Wrap buffer as cv::Mat with 4 dimensions: [Batch, Channels, Height, Width].
  // .clone() copies the data into an owning Mat so that `output` remains valid
  // if the thread_local buffer is reused on the next predict() call.
  int dims[] = {1, 3, this->input_height_, this->input_width_};
  output = cv::Mat(4, dims, CV_32F, local_buffer.data()).clone();
}

void RFDETRModel::postProcess(const std::vector<cv::Mat> &outputs,
                              int origin_height, int origin_width,
                              std::vector<Detection> &detections,
                              float confidence_threshold,
                              int max_number_boxes) {
  detections.clear();

  cv::Mat boxes = outputs[0];
  cv::Mat logits = outputs[1];
  cv::Mat masks;

  if (outputs.size() > 2) {
    masks = outputs[2];
  }

  boxes = boxes.reshape(1, {boxes.size[1], boxes.size[2]});
  logits = logits.reshape(1, {logits.size[1], logits.size[2]});

  int num_detections = boxes.rows;
  int num_classes = logits.cols;

  // Find the highest-scoring class per detection.
  // Optimization: since sigmoid is monotonic, the max logit equals the max
  // probability. We find the max logit with a single linear scan per row and
  // only apply sigmoid to that value, avoiding num_classes sigmoid calls.
  //
  // Thread-safety: thread_local scratch vectors avoid per-frame heap
  // allocations. Each thread has its own copies; resize() is O(1) when the
  // capacity is already sufficient (common case for a fixed model output size).
  thread_local std::vector<float> scores;
  thread_local std::vector<int> labels;
  thread_local std::vector<int> indices;
  scores.resize(num_detections);
  labels.resize(num_detections);
  indices.resize(num_detections);

  for (int i = 0; i < num_detections; ++i) {
    const float *row = logits.ptr<float>(i);
    float max_logit = row[0];
    int max_idx = 0;
    for (int j = 1; j < num_classes; ++j) {
      if (row[j] > max_logit) {
        max_logit = row[j];
        max_idx = j;
      }
    }
    scores[i] = 1.0f / (1.0f + std::exp(-max_logit));
    labels[i] = max_idx;
    indices[i] = i;
  }

  // Partial sort: O(N log K) vs O(N log N) full sort — significant when
  // max_number_boxes (K) is much smaller than num_detections (N, typically
  // 300).
  int num_to_keep = std::min(max_number_boxes, num_detections);
  std::partial_sort(indices.begin(), indices.begin() + num_to_keep,
                    indices.begin() + num_detections,
                    [&](int a, int b) { return scores[a] > scores[b]; });
  indices.resize(num_to_keep);

  for (int idx : indices) {
    float score = scores[idx];

    if (score <= confidence_threshold) {
      break; // All remaining scores are lower, no need to continue
    }

    Detection det;
    det.score = score;
    det.label = labels[idx];

    // Convert from cxcywh to xywh
    const float *box_row = boxes.ptr<float>(idx);
    float cx = box_row[0];
    float cy = box_row[1];
    float w = box_row[2];
    float h = box_row[3];

    float x_left = cx - w * 0.5f;
    float y_top = cy - h * 0.5f;

    det.normalizedBox = {x_left, y_top, w, h};
    det.unnormalizedBox = {x_left * origin_width, y_top * origin_height,
                           w * origin_width, h * origin_height};

    // Process mask if available
    if (!masks.empty()) {
      int maskHeight = masks.size[2];
      int maskWidth = masks.size[3];

      if (idx < masks.size[1]) {
        // Wrap mask (1, maskH, maskW)
        int maskDims[] = {maskHeight, maskWidth};
        cv::Mat maskRaw(2, maskDims, CV_32F,
                        const_cast<float *>(masks.ptr<float>(0, idx)));

        cv::Mat resizedMask;
        cv::resize(maskRaw, resizedMask, cv::Size(origin_width, origin_height));

        cv::Mat binaryMask;
        cv::threshold(resizedMask, binaryMask, 0.0, 255.0, cv::THRESH_BINARY);
        binaryMask.convertTo(det.mask, CV_8U);
      }
    }

    detections.push_back(det);
  }
}

void RFDETRModel::predict(const cv::Mat &image,
                          std::vector<Detection> &detections, Timings &timings,
                          float confidence_threshold, int max_number_boxes) {
  auto start_total = std::chrono::high_resolution_clock::now();

  int origin_height = image.rows;
  int origin_width = image.cols;

  // Preprocess
  auto start_pre = std::chrono::high_resolution_clock::now();
  cv::Mat input_tensor;
  this->preprocess(image, input_tensor);
  auto end_pre = std::chrono::high_resolution_clock::now();
  timings.preprocess =
      std::chrono::duration<float, std::milli>(end_pre - start_pre).count();

  // Run model
  auto start_run = std::chrono::high_resolution_clock::now();
  std::vector<cv::Mat> outputs;
  this->ort_session_->run(input_tensor, outputs);
  auto end_run = std::chrono::high_resolution_clock::now();
  timings.ort_run =
      std::chrono::duration<float, std::milli>(end_run - start_run).count();

  // Post-process
  auto start_post = std::chrono::high_resolution_clock::now();
  this->postProcess(outputs, origin_height, origin_width, detections,
                    confidence_threshold, max_number_boxes);
  auto end_post = std::chrono::high_resolution_clock::now();
  timings.postprocess =
      std::chrono::duration<float, std::milli>(end_post - start_post).count();

  auto end_total = std::chrono::high_resolution_clock::now();
  timings.total =
      std::chrono::duration<float, std::milli>(end_total - start_total).count();
}

void RFDETRModel::saveDetections(const cv::Mat &image,
                                 const std::vector<Detection> &detections,
                                 const std::string &save_path) const {
  cv::Mat result;
  drawDetections(image, detections, result);
  cv::imwrite(save_path, result);
}

} // namespace rfdetr
