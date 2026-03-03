#include "rfdetr_model.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

struct Stats {
  float mean;
  float median;
  float std;
  float min;
  float max;
};

Stats calculateStats(std::vector<float> times) {
  if (times.empty())
    return {0, 0, 0, 0, 0};

  float sum = std::accumulate(times.begin(), times.end(), 0.0f);
  float mean = sum / times.size();

  float sq_sum =
      std::inner_product(times.begin(), times.end(), times.begin(), 0.0f);
  float std = std::sqrt(sq_sum / times.size() - mean * mean);

  auto [min_it, max_it] = std::minmax_element(times.begin(), times.end());
  float min_val = *min_it;
  float max_val = *max_it;

  // Calculate median
  std::sort(times.begin(), times.end());
  float median;
  size_t size = times.size();
  if (size % 2 == 0) {
    median = (times[size / 2 - 1] + times[size / 2]) / 2;
  } else {
    median = times[size / 2];
  }

  return {mean, median, std, min_val, max_val};
}

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " <model_path> <input_path> <device> [iterations]"
              << std::endl;
    return 1;
  }

  std::string modelPath = argv[1];
  std::string inputPath = argv[2];
  std::string device = argv[3];
  int numIterations = (argc > 4) ? std::stoi(argv[4]) : 100;
  float sleepPerImage = (argc > 5) ? std::stof(argv[5]) : 0.0f;
  bool verbose = (argc > 6 && std::string(argv[6]) == "verbose");
  int warmupIterations = 10;

  std::cout << "Initializing model: " << modelPath << " on " << device
            << std::endl;
  rfdetr::RFDETRModel model(modelPath, device);

  std::vector<cv::Mat> images;

  // Check if input is a directory, video, or image
  if (fs::is_directory(inputPath)) {
    for (const auto &entry : fs::directory_iterator(inputPath)) {
      std::string ext = entry.path().extension().string();
      std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
      if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
        cv::Mat img = cv::imread(entry.path().string());
        if (!img.empty())
          images.push_back(img);
      }
    }
    if (images.empty()) {
      std::cerr << "Error: No images found in directory " << inputPath
                << std::endl;
      return 1;
    }
    std::cout << "Loaded " << images.size() << " images from directory."
              << std::endl;
  } else if (inputPath.find(".mp4") != std::string::npos ||
             inputPath.find(".avi") != std::string::npos) {
    cv::VideoCapture cap(inputPath);
    if (!cap.isOpened()) {
      std::cerr << "Error: Could not open video " << inputPath << std::endl;
      return 1;
    }
    cv::Mat frame;
    int count = 0;
    while (count < numIterations + warmupIterations && cap.read(frame)) {
      images.push_back(frame.clone());
      count++;
    }
    cap.release();
    std::cout << "Loaded " << images.size() << " frames from video."
              << std::endl;
  } else {
    cv::Mat img = cv::imread(inputPath);
    if (img.empty()) {
      std::cerr << "Error: Could not load image " << inputPath << std::endl;
      return 1;
    }
    images.push_back(img);
  }

  std::cout << "Running " << warmupIterations << " warmup iterations..."
            << std::endl;
  for (int i = 0; i < warmupIterations; ++i) {
    std::vector<rfdetr::Detection> dets;
    rfdetr::Timings timings;
    model.predict(images[i % images.size()], dets, timings);
  }

  std::cout << "Running " << numIterations << " benchmark iterations..."
            << std::endl;
  std::vector<float> pre_times, ort_times, post_times, total_processing_times;

  for (int i = 0; i < numIterations; ++i) {
    if (sleepPerImage > 0) {
      std::this_thread::sleep_for(
          std::chrono::milliseconds(static_cast<int>(sleepPerImage * 1000)));
    }
    std::vector<rfdetr::Detection> dets;
    rfdetr::Timings timings;
    model.predict(images[i % images.size()], dets, timings);

    float total = timings.preprocess + timings.ort_run + timings.postprocess;
    pre_times.push_back(timings.preprocess);
    ort_times.push_back(timings.ort_run);
    post_times.push_back(timings.postprocess);
    total_processing_times.push_back(total);

    if (verbose) {
      printf("Iteration %3d: Pre: %6.2fms, ORT: %6.2fms, Post: %6.2fms, Total: "
             "%6.2fms\n",
             i + 1, timings.preprocess, timings.ort_run, timings.postprocess,
             total);
    } else if ((i + 1) % 10 == 0) {
      std::cout << "Iteration " << (i + 1) << "/" << numIterations << std::endl;
    }
  }

  auto pre_stats = calculateStats(pre_times);
  auto ort_stats = calculateStats(ort_times);
  auto post_stats = calculateStats(post_times);
  auto total_stats = calculateStats(total_processing_times);

  std::string outputFilename = "benchmark_cpp_" + device + ".json";
  std::ofstream ofs(outputFilename);
  ofs << "{\n"
      << "  \"implementation\": \"C++\",\n"
      << "  \"device\": \"" << device << "\",\n"
      << "  \"num_iterations\": " << numIterations << ",\n"
      << "  \"data_source\": \"" << inputPath << "\",\n"
      << "  \"metrics\": {\n"
      << "    \"preprocessing\": {\"mean\": " << pre_stats.mean
      << ", \"median\": " << pre_stats.median << ", \"std\": " << pre_stats.std
      << ", \"min\": " << pre_stats.min << ", \"max\": " << pre_stats.max
      << "},\n"
      << "    \"ort_run\": {\"mean\": " << ort_stats.mean
      << ", \"median\": " << ort_stats.median << ", \"std\": " << ort_stats.std
      << ", \"min\": " << ort_stats.min << ", \"max\": " << ort_stats.max
      << "},\n"
      << "    \"postprocessing\": {\"mean\": " << post_stats.mean
      << ", \"median\": " << post_stats.median
      << ", \"std\": " << post_stats.std << ", \"min\": " << post_stats.min
      << ", \"max\": " << post_stats.max << "},\n"
      << "    \"total_processing\": {\"mean\": " << total_stats.mean
      << ", \"median\": " << total_stats.median
      << ", \"std\": " << total_stats.std << ", \"min\": " << total_stats.min
      << ", \"max\": " << total_stats.max
      << ", \"fps\": " << 1000.0 / total_stats.median << "}\n"
      << "  },\n"
      << "  \"iterations\": [\n";

  for (int i = 0; i < numIterations; ++i) {
    ofs << "    {\n"
        << "      \"index\": " << i + 1 << ",\n"
        << "      \"preprocessing\": " << pre_times[i] << ",\n"
        << "      \"ort_run\": " << ort_times[i] << ",\n"
        << "      \"postprocessing\": " << post_times[i] << ",\n"
        << "      \"total\": " << total_processing_times[i] << "\n"
        << "    }" << (i == numIterations - 1 ? "" : ",") << "\n";
  }

  ofs << "  ]\n"
      << "}\n";

  std::cout << "Results saved to " << outputFilename << std::endl;

  return 0;
}
