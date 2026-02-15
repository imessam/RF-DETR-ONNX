#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace rfdetr {

/**
 * @brief Apply sigmoid activation function element-wise
 * @param x Input matrix
 * @param output Output matrix with sigmoid applied
 */
void sigmoid(const cv::Mat& x, cv::Mat& output);

/**
 * @brief Convert bounding boxes from center-xy-width-height to min-max normalized format
 * @param boxes Input boxes in cxcywh format (N x 4)
 * @param output Output boxes in xyxyn format (N x 4)
 */
void boxCxcywhToXyxyn(const cv::Mat& boxes, cv::Mat& output);

} // namespace rfdetr
