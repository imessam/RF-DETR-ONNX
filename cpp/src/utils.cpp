#include "utils.hpp"
#include <cmath>

namespace rfdetr {

void sigmoid(const cv::Mat &x, cv::Mat &output) {
  // Optimized sigmoid using raw pointer loop
  output.create(x.dims, x.size.p, x.type());

  const float *src = reinterpret_cast<const float *>(x.data);
  float *dst = reinterpret_cast<float *>(output.data);
  size_t total = x.total();

  for (size_t i = 0; i < total; ++i) {
    dst[i] = 1.0f / (1.0f + std::exp(-src[i]));
  }
}

void boxCxcywhToXyxyn(const cv::Mat &boxes, cv::Mat &output) {
  // Vectorized box conversion using raw pointer arithmetic
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

} // namespace rfdetr
