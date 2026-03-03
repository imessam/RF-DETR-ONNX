#pragma once

#include "rfdetr_model.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

namespace rfdetr {

/**
 * @brief Convert bounding boxes from center-xy-width-height to min-max
 * normalized format.
 * @param boxes  Input boxes in cxcywh format (N x 4).
 * @param output Output boxes in xyxyn format (N x 4).
 */
void boxCxcywhToXyxyn(const cv::Mat &boxes, cv::Mat &output);

/**
 * @brief Convert bounding boxes from center-xy-width-height to xywh normalized
 * format.
 * @param boxes  Input boxes in cxcywh format (N x 4).
 * @param output Output boxes in xywh format (N x 4).
 */
void boxCxcywhToXywh(const cv::Mat &boxes, cv::Mat &output);

/**
 * @brief Draw bounding boxes and class labels onto an image.
 * @param image     Source image in BGR format (not modified).
 * @param detections Detection results to visualize.
 * @param output    Destination image (will be a clone of image with overlays).
 * @param fps       Optional FPS overlay; pass a negative value to skip.
 */
void drawDetections(const cv::Mat &image,
                    const std::vector<Detection> &detections, cv::Mat &output,
                    double fps = -1.0);

} // namespace rfdetr
