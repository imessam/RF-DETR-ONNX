#include "utils.hpp"

namespace rfdetr {

void boxCxcywhToXyxyn(const cv::Mat &boxes, cv::Mat &output) {
  output.create(boxes.size(), boxes.type());

  const float *src = boxes.ptr<float>();
  float *dst = output.ptr<float>();
  int numBoxes = boxes.rows;

  for (int i = 0; i < numBoxes; ++i) {
    float cx = src[i * 4 + 0];
    float cy = src[i * 4 + 1];
    float hw = src[i * 4 + 2] * 0.5f;
    float hh = src[i * 4 + 3] * 0.5f;

    dst[i * 4 + 0] = cx - hw; // xmin
    dst[i * 4 + 1] = cy - hh; // ymin
    dst[i * 4 + 2] = cx + hw; // xmax
    dst[i * 4 + 3] = cy + hh; // ymax
  }
}

void boxCxcywhToXywh(const cv::Mat &boxes, cv::Mat &output) {
  output.create(boxes.size(), boxes.type());

  const float *src = boxes.ptr<float>();
  float *dst = output.ptr<float>();
  int numBoxes = boxes.rows;

  for (int i = 0; i < numBoxes; ++i) {
    float cx = src[i * 4 + 0];
    float cy = src[i * 4 + 1];
    float w = src[i * 4 + 2];
    float h = src[i * 4 + 3];

    dst[i * 4 + 0] = cx - w * 0.5f; // x_left
    dst[i * 4 + 1] = cy - h * 0.5f; // y_top
    dst[i * 4 + 2] = w;
    dst[i * 4 + 3] = h;
  }
}

} // namespace rfdetr
