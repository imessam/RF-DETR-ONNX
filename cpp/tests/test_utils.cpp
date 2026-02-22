#include "utils.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

namespace rfdetr {

TEST(UtilsTest, SigmoidTest) {
  cv::Mat input = (cv::Mat_<float>(1, 3) << -100.0f, 0.0f, 100.0f);
  cv::Mat output;
  sigmoid(input, output);

  EXPECT_NEAR(output.at<float>(0, 0), 0.0f, 1e-6);
  EXPECT_NEAR(output.at<float>(0, 1), 0.5f, 1e-6);
  EXPECT_NEAR(output.at<float>(0, 2), 1.0f, 1e-6);
}

TEST(UtilsTest, BoxConversionTest) {
  // Input: cx, cy, w, h
  cv::Mat input = (cv::Mat_<float>(1, 4) << 0.5f, 0.5f, 0.2f, 0.4f);
  cv::Mat output;
  boxCxcywhToXyxyn(input, output);

  // Expected: xmin = 0.5 - 0.1 = 0.4
  //           ymin = 0.5 - 0.2 = 0.3
  //           xmax = 0.5 + 0.1 = 0.6
  //           ymax = 0.5 + 0.2 = 0.7
  EXPECT_NEAR(output.at<float>(0, 0), 0.4f, 1e-6);
  EXPECT_NEAR(output.at<float>(0, 1), 0.3f, 1e-6);
  EXPECT_NEAR(output.at<float>(0, 2), 0.6f, 1e-6);
  EXPECT_NEAR(output.at<float>(0, 3), 0.7f, 1e-6);
}

TEST(UtilsTest, BoxConversionToXywhTest) {
  // Input: cx, cy, w, h
  cv::Mat input = (cv::Mat_<float>(1, 4) << 0.5f, 0.5f, 0.2f, 0.4f);
  cv::Mat output;
  boxCxcywhToXywh(input, output);

  // Expected: x_left = 0.5 - 0.1 = 0.4
  //           y_top = 0.5 - 0.2 = 0.3
  //           w = 0.2
  //           h = 0.4
  EXPECT_NEAR(output.at<float>(0, 0), 0.4f, 1e-6);
  EXPECT_NEAR(output.at<float>(0, 1), 0.3f, 1e-6);
  EXPECT_NEAR(output.at<float>(0, 2), 0.2f, 1e-6);
  EXPECT_NEAR(output.at<float>(0, 3), 0.4f, 1e-6);
}

} // namespace rfdetr

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
