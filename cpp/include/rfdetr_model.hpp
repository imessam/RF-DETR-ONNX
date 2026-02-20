#pragma once

#include "onnx_runtime.hpp"
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace rfdetr {

// Constants
constexpr float DEFAULT_CONFIDENCE_THRESHOLD = 0.5f;
constexpr int DEFAULT_MAX_NUMBER_BOXES = 300;

/**
 * @brief Structure to hold a single detection result
 */
struct Detection {
  float score;
  int label;
  cv::Rect2f normalizedBox;   // [x, y, w, h] normalized [0, 1]
  cv::Rect2f unnormalizedBox; // [x, y, w, h] in pixels
  cv::Mat mask;
};

/**
 * @brief Structure to hold timing information
 */
struct Timings {
  float preprocess = 0.0f;
  float ortRun = 0.0f;
  float postprocess = 0.0f;
  float total = 0.0f;
};

/**
 * @brief High-level class for RF-DETR model inference
 *
 * Handles the complete inference pipeline including preprocessing,
 * model execution, and postprocessing with visualization.
 */
class RFDETRModel {
public:
  /**
   * @brief Initialize the RF-DETR model
   * @param modelPath Path to the ONNX model file
   * @param device Device preference ("gpu" or "cpu")
   */
  RFDETRModel(const std::string &modelPath, const std::string &device = "gpu");

  /**
   * @brief Perform a warmup run to initialize GPU/TensorRT resources
   */
  void warmup();

  /**
   * @brief Run model inference and return detections
   * @param image Input image in BGR format (H, W, C)
   * @param detections Output detection results
   * @param timings Output timing information
   * @param confidenceThreshold Confidence threshold for filtering
   * @param maxNumberBoxes Maximum number of boxes to return
   */
  void predict(const cv::Mat &image, std::vector<Detection> &detections,
               Timings &timings,
               float confidenceThreshold = DEFAULT_CONFIDENCE_THRESHOLD,
               int maxNumberBoxes = DEFAULT_MAX_NUMBER_BOXES);

  /**
   * @brief Draw bounding boxes, masks and labels on image and save
   * @param image Original image in BGR format
   * @param detections Detection results to visualize
   * @param savePath Path to save the output image
   */
  void saveDetections(const cv::Mat &image,
                      const std::vector<Detection> &detections,
                      const std::string &savePath) const;

private:
  /**
   * @brief Preprocess the input image for inference
   * @param image Input image in BGR format (H, W, C)
   * @param output Preprocessed image batch (1, C, H, W)
   */
  void preprocess(const cv::Mat &image, cv::Mat &output);

  /**
   * @brief Post-process model outputs to extract detections
   * @param outputs Raw model outputs
   * @param originHeight Original image height
   * @param originWidth Original image width
   * @param detections Output detection results
   * @param confidenceThreshold Confidence threshold for filtering
   * @param maxNumberBoxes Maximum number of boxes to return
   */
  void postProcess(const std::vector<cv::Mat> &outputs, int originHeight,
                   int originWidth, std::vector<Detection> &detections,
                   float confidenceThreshold, int maxNumberBoxes);

  std::unique_ptr<OnnxRuntimeSession> ortSession_;
  int inputHeight_;
  int inputWidth_;

  // Pre-allocated buffer for preprocessing output (avoids allocation per call)
  std::vector<float> preprocessBuffer_;

  // Pre-computed normalization: pixel * normScale_[c] + normOffset_[c]
  // where normScale = 1.0 / (255.0 * std), normOffset = -mean / std
  float normScale_[3];
  float normOffset_[3];

  static constexpr float MEANS[3] = {0.485f, 0.456f, 0.406f};
  static constexpr float STDS[3] = {0.229f, 0.224f, 0.225f};
};

} // namespace rfdetr
