#include "rfdetr_model.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <random>

namespace rfdetr {

RFDETRModel::RFDETRModel(const std::string &modelPath,
                         const std::string &device) {
  // Initialize ONNX Runtime session
  ortSession_ = std::make_unique<OnnxRuntimeSession>(modelPath, device);

  // Get input shape
  auto inputShape = ortSession_->getInputShape();
  inputHeight_ = static_cast<int>(inputShape[2]);
  inputWidth_ = static_cast<int>(inputShape[3]);

  // Pre-allocate preprocessing buffer (1 * 3 * H * W)
  preprocessBuffer_.resize(3 * inputHeight_ * inputWidth_);

  // Pre-compute fused normalization constants:
  //   normalized = (pixel/255.0 - mean) / std
  //             = pixel * (1.0 / (255.0 * std)) + (-mean / std)
  for (int i = 0; i < 3; ++i) {
    normScale_[i] = 1.0f / (255.0f * STDS[i]);
    normOffset_[i] = -MEANS[i] / STDS[i];
  }

  // Perform warmup
  warmup();
}

void RFDETRModel::warmup() { ortSession_->warmup(); }

void RFDETRModel::preprocess(const cv::Mat &image, cv::Mat &output) {
  // Convert BGR to RGB first (on original image)
  cv::Mat rgb;
  cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);

  // Resize to model input size
  cv::Mat resized;
  cv::resize(rgb, resized, cv::Size(inputWidth_, inputHeight_));

  // Use OpenCV's SIMD-optimized convertTo + split, then normalize per-channel
  const int planeSize = inputHeight_ * inputWidth_;

  // Convert uint8 to float32 normalized to [0,1] (SIMD-optimized)
  cv::Mat floatRgb;
  resized.convertTo(floatRgb, CV_32FC3, 1.0 / 255.0);

  // Split into separate channels
  cv::Mat channels[3];
  cv::split(floatRgb, channels);

  // Normalize each channel and copy into pre-allocated CHW buffer
  float *bufPtr = preprocessBuffer_.data();
  for (int c = 0; c < 3; ++c) {
    channels[c] = (channels[c] - MEANS[c]) / STDS[c];
    std::memcpy(bufPtr + c * planeSize, channels[c].data,
                planeSize * sizeof(float));
  }

  // Wrap buffer as cv::Mat (no copy, shares data with preprocessBuffer_)
  int dims[] = {1, 3, inputHeight_, inputWidth_};
  output = cv::Mat(4, dims, CV_32F, preprocessBuffer_.data());
}

void RFDETRModel::postProcess(const std::vector<cv::Mat> &outputs,
                              int originHeight, int originWidth,
                              Detection &detection, float confidenceThreshold,
                              int maxNumberBoxes) {
  detection.clear();

  // outputs[0]: boxes (1, N, 4)
  // outputs[1]: scores/logits (1, N, num_classes)
  cv::Mat boxes = outputs[0];
  cv::Mat logits = outputs[1];
  cv::Mat masks;

  if (outputs.size() > 2) {
    masks = outputs[2];
  }

  // Reshape for easier processing: remove batch dimension
  boxes = boxes.reshape(1, {boxes.size[1], boxes.size[2]});
  logits = logits.reshape(1, {logits.size[1], logits.size[2]});

  int numDetections = boxes.rows;
  int numClasses = logits.cols;

  // Fused sigmoid + max-score scan: compute sigmoid only for the max logit
  // per detection row, avoiding the full N×80 sigmoid computation
  std::vector<float> scores(numDetections);
  std::vector<int> labels(numDetections);
  std::vector<int> indices(numDetections);

  for (int i = 0; i < numDetections; ++i) {
    const float *row = logits.ptr<float>(i);
    float maxLogit = row[0];
    int maxIdx = 0;
    for (int j = 1; j < numClasses; ++j) {
      if (row[j] > maxLogit) {
        maxLogit = row[j];
        maxIdx = j;
      }
    }
    // Apply sigmoid only to the max logit (monotonic, so max logit = max prob)
    scores[i] = 1.0f / (1.0f + std::exp(-maxLogit));
    labels[i] = maxIdx;
    indices[i] = i;
  }

  // Sort by score descending
  std::sort(indices.begin(), indices.end(),
            [&scores](int a, int b) { return scores[a] > scores[b]; });

  // Limit to maxNumberBoxes
  int numToKeep = std::min(maxNumberBoxes, static_cast<int>(indices.size()));
  indices.resize(numToKeep);

  // Filter by confidence and convert only passing boxes (deferred conversion)
  // Early termination: since indices are sorted by score, stop when below
  // threshold
  for (int idx : indices) {
    float score = scores[idx];

    if (score <= confidenceThreshold) {
      break; // All remaining scores are lower, no need to continue
    }

    detection.scores.push_back(score);
    detection.labels.push_back(labels[idx]);

    // Convert this single box from cxcywh to xyxy and scale to image size
    const float *boxRow = boxes.ptr<float>(idx);
    float cx = boxRow[0];
    float cy = boxRow[1];
    float hw = boxRow[2] * 0.5f;
    float hh = boxRow[3] * 0.5f;

    float xmin = (cx - hw) * originWidth;
    float ymin = (cy - hh) * originHeight;
    float xmax = (cx + hw) * originWidth;
    float ymax = (cy + hh) * originHeight;

    detection.boxes.emplace_back(xmin, ymin, xmax - xmin, ymax - ymin);
  }

  // Process masks if available
  if (!masks.empty() && !detection.boxes.empty()) {
    masks = masks.reshape(1, {masks.size[1], masks.size[2], masks.size[3]});

    int maskHeight = masks.size[1];
    int maskWidth = masks.size[2];

    for (size_t i = 0; i < detection.boxes.size(); ++i) {
      int originalIdx = indices[i];

      if (originalIdx < masks.size[0]) {
        cv::Mat mask(maskHeight, maskWidth, CV_32F,
                     masks.ptr<float>(originalIdx));

        cv::Mat resizedMask;
        cv::resize(mask, resizedMask, cv::Size(originWidth, originHeight));

        cv::Mat binaryMask;
        cv::threshold(resizedMask, binaryMask, 0.0, 255.0, cv::THRESH_BINARY);
        binaryMask.convertTo(binaryMask, CV_8U);

        detection.masks.push_back(binaryMask);
      }
    }
  }
}

void RFDETRModel::predict(const cv::Mat &image, Detection &detection,
                          Timings &timings, float confidenceThreshold,
                          int maxNumberBoxes) {
  auto startTotal = std::chrono::high_resolution_clock::now();

  int originHeight = image.rows;
  int originWidth = image.cols;

  // Preprocess
  auto startPre = std::chrono::high_resolution_clock::now();
  cv::Mat inputTensor;
  preprocess(image, inputTensor);
  auto endPre = std::chrono::high_resolution_clock::now();
  timings.preprocess =
      std::chrono::duration<float, std::milli>(endPre - startPre).count();

  // Run model
  auto startRun = std::chrono::high_resolution_clock::now();
  std::vector<cv::Mat> outputs;
  ortSession_->run(inputTensor, outputs);
  auto endRun = std::chrono::high_resolution_clock::now();
  timings.ortRun =
      std::chrono::duration<float, std::milli>(endRun - startRun).count();

  // Post-process
  auto startPost = std::chrono::high_resolution_clock::now();
  postProcess(outputs, originHeight, originWidth, detection,
              confidenceThreshold, maxNumberBoxes);
  auto endPost = std::chrono::high_resolution_clock::now();
  timings.postprocess =
      std::chrono::duration<float, std::milli>(endPost - startPost).count();

  auto endTotal = std::chrono::high_resolution_clock::now();
  timings.total =
      std::chrono::duration<float, std::milli>(endTotal - startTotal).count();
}

void RFDETRModel::saveDetections(const cv::Mat &image,
                                 const Detection &detection,
                                 const std::string &savePath) const {
  // Create a copy to draw on
  cv::Mat result = image.clone();

  // Generate random colors for each unique label
  std::map<int, cv::Scalar> labelColors;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 255);

  for (int label : detection.labels) {
    if (labelColors.find(label) == labelColors.end()) {
      labelColors[label] = cv::Scalar(dis(gen), dis(gen), dis(gen));
    }
  }

  // Draw masks with transparency if available
  if (!detection.masks.empty()) {
    cv::Mat overlay = result.clone();

    for (size_t i = 0; i < detection.masks.size(); ++i) {
      int label = detection.labels[i];
      cv::Scalar color = labelColors[label];

      cv::Mat colorMask(detection.masks[i].size(), CV_8UC3);
      colorMask.setTo(color, detection.masks[i]);

      cv::addWeighted(overlay, 1.0, colorMask, 0.4, 0.0, overlay);
    }

    cv::addWeighted(result, 0.6, overlay, 0.4, 0.0, result);
  }

  // Draw bounding boxes and labels
  for (size_t i = 0; i < detection.boxes.size(); ++i) {
    const auto &box = detection.boxes[i];
    int label = detection.labels[i];
    cv::Scalar color = labelColors[label];

    cv::rectangle(result, box, color, 4);

    std::string text = std::to_string(label);
    int baseline = 0;
    cv::Size textSize =
        cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline);

    cv::Point textOrg(box.x + 5, box.y + textSize.height + 5);
    cv::putText(result, text, textOrg, cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
  }

  // Save result
  cv::imwrite(savePath, result);
}

} // namespace rfdetr
