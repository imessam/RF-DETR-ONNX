#include "rfdetr_model.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace fs = std::filesystem;

void saveDetectionsJson(const std::string &outputPath, const std::string &assetName,
                        const rfdetr::Detection &det, float latencyMs) {
  std::ofstream ofs(outputPath);
  ofs << "{\n"
      << "  \"asset\": \"" << assetName << "\",\n"
      << "  \"implementation\": \"C++ ONNX\",\n"
      << "  \"latency_ms\": " << latencyMs << ",\n"
      << "  \"detections\": [\n";

  for (size_t i = 0; i < det.boxes.size(); ++i) {
    ofs << "    {\n"
        << "      \"bbox\": [" << det.boxes[i].x << ", " << det.boxes[i].y << ", "
        << det.boxes[i].x + det.boxes[i].width << ", "
        << det.boxes[i].y + det.boxes[i].height << "],\n"
        << "      \"class_id\": " << det.labels[i] << ",\n"
        << "      \"score\": " << det.scores[i] << "\n"
        << "    }" << (i == det.boxes.size() - 1 ? "" : ",") << "\n";
  }

  ofs << "  ]\n"
      << "}\n";
}

int main(int argc, char **argv) {
  if (argc < 5) {
    std::cerr << "Usage: " << argv[0]
              << " <model_path> <input_dir> <device> <output_dir>"
              << std::endl;
    return 1;
  }

  std::string modelPath = argv[1];
  std::string inputDir = argv[2];
  std::string device = argv[3];
  std::string outputDir = argv[4];

  std::cout << "Initializing C++ ONNX Model on " << device << "..." << std::endl;
  rfdetr::RFDETRModel model(modelPath, device);

  if (!fs::exists(outputDir)) {
    fs::create_directories(outputDir);
  }

  for (const auto &entry : fs::directory_iterator(inputDir)) {
    std::string ext = entry.path().extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
      std::string assetName = entry.path().filename().string();
      cv::Mat img = cv::imread(entry.path().string());
      if (img.empty()) continue;

      rfdetr::Detection det;
      rfdetr::Timings timings;
      
      auto start = std::chrono::high_resolution_clock::now();
      model.predict(img, det, timings);
      auto end = std::chrono::high_resolution_clock::now();
      float latency = std::chrono::duration<float, std::milli>(end - start).count();

      std::string baseName = entry.path().stem().string();
      std::string jsonPath = outputDir + "/" + baseName + ".json";
      
      saveDetectionsJson(jsonPath, assetName, det, latency);
      std::cout << "  - " << assetName << ": " << det.boxes.size() 
                << " detections (" << latency << " ms)" << std::endl;
    }
  }

  return 0;
}
