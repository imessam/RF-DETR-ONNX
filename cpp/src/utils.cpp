#include "utils.hpp"

namespace rfdetr {

void sigmoid(const cv::Mat &x, cv::Mat &output) {
  // Sigmoid: 1 / (1 + exp(-x))
  cv::Mat negX;
  cv::exp(-x, negX);
  output = 1.0f / (1.0f + negX);
}

void boxCxcywhToXyxyn(const cv::Mat &boxes, cv::Mat &output) {
  // Input: boxes in format [center_x, center_y, width, height]
  // Output: boxes in format [xmin, ymin, xmax, ymax]

  output.create(boxes.size(), boxes.type());

  for (int i = 0; i < boxes.rows; ++i) {
    float cx = boxes.at<float>(i, 0);
    float cy = boxes.at<float>(i, 1);
    float w = boxes.at<float>(i, 2);
    float h = boxes.at<float>(i, 3);

    output.at<float>(i, 0) = cx - w / 2.0f; // xmin
    output.at<float>(i, 1) = cy - h / 2.0f; // ymin
    output.at<float>(i, 2) = cx + w / 2.0f; // xmax
    output.at<float>(i, 3) = cy + h / 2.0f; // ymax
  }
}

} // namespace rfdetr
