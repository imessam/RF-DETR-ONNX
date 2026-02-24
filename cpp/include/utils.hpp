#pragma once

#include <opencv2/opencv.hpp>

namespace rfdetr {

/**
 * @brief Convert bounding boxes from center-xy-width-height to min-max
 * normalized format
 * @param boxes Input boxes in cxcywh format (N x 4)
 * @param output Output boxes in xyxyn format (N x 4)
 */
void boxCxcywhToXyxyn(const cv::Mat &boxes, cv::Mat &output);

/**
 * @brief Convert bounding boxes from center-xy-width-height to xywh normalized
 * format
 * @param boxes Input boxes in cxcywh format (N x 4)
 * @param output Output boxes in xywh format (N x 4)
 */
void boxCxcywhToXywh(const cv::Mat &boxes, cv::Mat &output);

} // namespace rfdetr
