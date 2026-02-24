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

  // Get input shape from the session
  auto inputShape = ortSession_->getInputShape();
  inputHeight_ = static_cast<int>(inputShape[2]);
  inputWidth_ = static_cast<int>(inputShape[3]);

  // Pre-allocate the preprocessing buffer once (1 × 3 × H × W floats)
  preprocessBuffer_.resize(3 * inputHeight_ * inputWidth_);

  // Perform a warmup run to initialize GPU resources
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

  const int planeSize = inputHeight_ * inputWidth_;

  // Convert to float in [0, 1]
  cv::Mat floatRgb;
  resized.convertTo(floatRgb, CV_32FC3, 1.0 / 255.0);

  // Split into per-channel planes
  cv::Mat channels[3];
  cv::split(floatRgb, channels);

  // Normalize each channel with ImageNet mean/std and copy into CHW buffer
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
                              std::vector<Detection> &detections,
                              float confidenceThreshold, int maxNumberBoxes) {
  detections.clear();

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

  // Find the highest-scoring class per detection.
  // Optimization: since sigmoid is monotonic, the max logit equals the max
  // probability. We find the max logit with a single linear scan per row and
  // only apply sigmoid to that value, avoiding num_classes sigmoid calls.
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

    Detection det;
    det.score = score;
    det.label = labels[idx];

    // Convert from cxcywh to xywh
    const float *boxRow = boxes.ptr<float>(idx);
    float cx = boxRow[0];
    float cy = boxRow[1];
    float w = boxRow[2];
    float h = boxRow[3];

    float x_left = cx - w * 0.5f;
    float y_top = cy - h * 0.5f;

    det.normalizedBox = cv::Rect2f(x_left, y_top, w, h);
    det.unnormalizedBox = cv::Rect2f(x_left * originWidth, y_top * originHeight,
                                     w * originWidth, h * originHeight);

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
        cv::resize(maskRaw, resizedMask, cv::Size(originWidth, originHeight));

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
                          float confidenceThreshold, int maxNumberBoxes) {
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
  postProcess(outputs, originHeight, originWidth, detections,
              confidenceThreshold, maxNumberBoxes);
  auto endPost = std::chrono::high_resolution_clock::now();
  timings.postprocess =
      std::chrono::duration<float, std::milli>(endPost - startPost).count();

  auto endTotal = std::chrono::high_resolution_clock::now();
  timings.total =
      std::chrono::duration<float, std::milli>(endTotal - startTotal).count();
}

void RFDETRModel::saveDetections(const cv::Mat &image,
                                 const std::vector<Detection> &detections,
                                 const std::string &savePath) const {
  // Create a copy to draw on
  cv::Mat result = image.clone();

  // Generate a random color per unique label.
  // Note: colors are re-randomized on every call. For consistent per-label
  // colors across multiple images, seed with a fixed value or use a label hash.
  std::map<int, cv::Scalar> labelColors;
  std::mt19937 gen(std::random_device{}());
  std::uniform_int_distribution<> dis(0, 255);

  for (const auto &det : detections) {
    if (labelColors.find(det.label) == labelColors.end()) {
      labelColors[det.label] = cv::Scalar(dis(gen), dis(gen), dis(gen));
    }
  }

  // Draw semi-transparent masks if the model produced segmentation output
  bool hasMasks =
      std::any_of(detections.begin(), detections.end(),
                  [](const Detection &d) { return !d.mask.empty(); });

  if (hasMasks) {
    cv::Mat overlay = result.clone();

    for (const auto &det : detections) {
      if (det.mask.empty())
        continue;

      cv::Scalar color = labelColors[det.label];
      overlay.setTo(color, det.mask);
    }

    cv::addWeighted(overlay, 0.5, result, 0.5, 0.0, result);
  }

  // Draw bounding boxes and labels
  for (const auto &det : detections) {
    const auto &box = det.unnormalizedBox;
    cv::Scalar color = labelColors[det.label];

    cv::rectangle(result, box, color, 4);

    std::string text = std::to_string(det.label);
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
