#include "utils.hpp"
#include <string>

namespace rfdetr {

void drawDetections(const cv::Mat &image,
                    const std::vector<detectiondata::Detection> &detections,
                    cv::Mat &output, double fps) {
  output = image.clone();

  // Derive a stable, visually distinct color per class using an integer hash.
  // Knuth multiplicative hash — no RNG, no allocation, no color flickering.
  auto class_color = [](int id) -> cv::Scalar {
    uint32_t h = static_cast<uint32_t>(id) * 2654435761u;
    return cv::Scalar((h & 0xFF), ((h >> 8) & 0xFF), ((h >> 16) & 0xFF));
  };

  for (const auto &det : detections) {
    const auto &box = det.box;
    cv::Scalar color = class_color(det.class_id);

    cv::rectangle(output, cv::Rect(box.x, box.y, box.width, box.height), color,
                  4);

    std::string text = std::to_string(det.class_id);
    int baseline = 0;
    cv::Size text_size =
        cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline);
    cv::Point text_org(box.x + 5, box.y + text_size.height + 5);
    cv::putText(output, text, text_org, cv::FONT_HERSHEY_SIMPLEX, 0.7, color,
                2);
  }

  if (fps >= 0.0) {
    std::string fps_text = "FPS: " + std::to_string(fps).substr(0, 5);
    cv::putText(output, fps_text, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX,
                1.2, cv::Scalar(0, 255, 0), 2);
  }
}

void boxCxcywhToXyxyn(const cv::Mat &boxes, cv::Mat &output) {
  output.create(boxes.size(), boxes.type());

  const float *src = boxes.ptr<float>();
  float *dst = output.ptr<float>();
  int num_boxes = boxes.rows;

  for (int i = 0; i < num_boxes; ++i) {
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
  int num_boxes = boxes.rows;

  for (int i = 0; i < num_boxes; ++i) {
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
