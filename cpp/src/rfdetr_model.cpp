#include "rfdetr_model.hpp"
#include "utils.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
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

  // Pre-convert normalization constants (C, 1, 1)
  means_ = cv::Mat(3, 1, CV_32F);
  stds_ = cv::Mat(3, 1, CV_32F);

  for (int i = 0; i < 3; ++i) {
    means_.at<float>(i, 0) = MEANS[i];
    stds_.at<float>(i, 0) = STDS[i];
  }

  // Reshape for broadcasting (1, 3, 1, 1)
  means_ = means_.reshape(1, {1, 3, 1, 1});
  stds_ = stds_.reshape(1, {1, 3, 1, 1});

  // Perform warmup
  warmup();
}

void RFDETRModel::warmup() { ortSession_->warmup(); }

void RFDETRModel::preprocess(const cv::Mat &image, cv::Mat &output) {
  // Convert BGR to RGB
  cv::Mat rgb;
  cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);

  // Resize to model input size
  cv::Mat resized;
  cv::resize(rgb, resized, cv::Size(inputWidth_, inputHeight_));

  // Convert to float32 and normalize to [0, 1]
  cv::Mat float_img;
  resized.convertTo(float_img, CV_32F, 1.0 / 255.0);

  // Convert from HWC to CHW
  std::vector<cv::Mat> channels(3);
  cv::split(float_img, channels);

  // Stack channels and add batch dimension: (1, 3, H, W)
  output = cv::Mat(1, 3 * inputHeight_ * inputWidth_, CV_32F);

  for (int c = 0; c < 3; ++c) {
    // Normalize each channel: (img - mean) / std
    cv::Mat normalized = (channels[c] - MEANS[c]) / STDS[c];

    // Copy to output tensor
    std::memcpy(output.ptr<float>() + c * inputHeight_ * inputWidth_,
                normalized.data, inputHeight_ * inputWidth_ * sizeof(float));
  }

  // Reshape to (1, 3, H, W)
  int dims[] = {1, 3, inputHeight_, inputWidth_};
  output = output.reshape(1, 4, dims);
}

void RFDETRModel::postProcess(const std::vector<cv::Mat> &outputs,
                              int originHeight, int originWidth,
                              Detection &detection, float confidenceThreshold,
                              int maxNumberBoxes) {
  detection.clear();

  // outputs[0]: boxes (1, N, 4)
  // outputs[1]: scores/logits (1, N, num_classes)
  // outputs[2]: masks (optional) (1, N, mask_h, mask_w)

  cv::Mat boxes = outputs[0];
  cv::Mat logits = outputs[1];
  cv::Mat masks;

  if (outputs.size() > 2) {
    masks = outputs[2];
  }

  // Apply sigmoid to logits to get probabilities
  cv::Mat prob;
  sigmoid(logits, prob);

  // Reshape for easier processing: remove batch dimension
  // boxes: (N, 4), prob: (N, num_classes)
  boxes = boxes.reshape(1, {boxes.size[1], boxes.size[2]});
  prob = prob.reshape(1, {prob.size[1], prob.size[2]});

  int numDetections = boxes.rows;
  int numClasses = prob.cols;

  // Get max confidence and corresponding label for each detection
  std::vector<float> scores;
  std::vector<int> labels;
  std::vector<int> indices;

  for (int i = 0; i < numDetections; ++i) {
    double maxScore;
    cv::Point maxLoc;
    cv::minMaxLoc(prob.row(i), nullptr, &maxScore, nullptr, &maxLoc);

    scores.push_back(static_cast<float>(maxScore));
    labels.push_back(maxLoc.x);
    indices.push_back(i);
  }

  // Sort by score descending
  std::sort(indices.begin(), indices.end(),
            [&scores](int a, int b) { return scores[a] > scores[b]; });

  // Limit to maxNumberBoxes
  int numToKeep = std::min(maxNumberBoxes, static_cast<int>(indices.size()));
  indices.resize(numToKeep);

  // Convert boxes from cxcywh to xyxyn format
  cv::Mat boxesXyxy;
  boxCxcywhToXyxyn(boxes, boxesXyxy);

  // Scale boxes to original image size and filter by confidence
  for (int idx : indices) {
    float score = scores[idx];

    if (score > confidenceThreshold) {
      detection.scores.push_back(score);
      detection.labels.push_back(labels[idx]);

      // Scale boxes to original size
      float xmin = boxesXyxy.at<float>(idx, 0) * originWidth;
      float ymin = boxesXyxy.at<float>(idx, 1) * originHeight;
      float xmax = boxesXyxy.at<float>(idx, 2) * originWidth;
      float ymax = boxesXyxy.at<float>(idx, 3) * originHeight;

      detection.boxes.emplace_back(xmin, ymin, xmax - xmin, ymax - ymin);
    }
  }

  // Process masks if available
  if (!masks.empty() && !detection.boxes.empty()) {
    // Reshape masks: (N, mask_h, mask_w)
    masks = masks.reshape(1, {masks.size[1], masks.size[2], masks.size[3]});

    int maskHeight = masks.size[1];
    int maskWidth = masks.size[2];

    for (size_t i = 0; i < detection.boxes.size(); ++i) {
      int originalIdx = indices[i];

      if (originalIdx < masks.size[0]) {
        // Extract mask for this detection
        cv::Mat mask(maskHeight, maskWidth, CV_32F,
                     masks.ptr<float>(originalIdx));

        // Resize to original image size
        cv::Mat resizedMask;
        cv::resize(mask, resizedMask, cv::Size(originWidth, originHeight));

        // Threshold and convert to binary mask
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

      // Create colored mask
      cv::Mat colorMask(detection.masks[i].size(), CV_8UC3);
      colorMask.setTo(color, detection.masks[i]);

      // Blend with overlay
      cv::addWeighted(overlay, 1.0, colorMask, 0.4, 0.0, overlay);
    }

    // Blend overlay with original image
    cv::addWeighted(result, 0.6, overlay, 0.4, 0.0, result);
  }

  // Draw bounding boxes and labels
  for (size_t i = 0; i < detection.boxes.size(); ++i) {
    const auto &box = detection.boxes[i];
    int label = detection.labels[i];
    cv::Scalar color = labelColors[label];

    // Draw rectangle
    cv::rectangle(result, box, color, 4);

    // Draw label text
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
