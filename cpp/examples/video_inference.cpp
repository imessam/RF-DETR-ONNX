#include "rfdetr_model.hpp"
#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <random>
#include <string>

void printUsage(const char *programName) {
  std::cout
      << "Usage: " << programName << " [OPTIONS]\n"
      << "Run video inference with a RF-DETR ONNX model.\n\n"
      << "Required arguments:\n"
      << "  --model PATH        Path to the ONNX model file\n"
      << "  --video PATH        Path to the input video\n\n"
      << "Optional arguments:\n"
      << "  --output PATH       Path to save output video (default: "
         "../output/output.mp4)\n"
      << "  --threshold FLOAT   Confidence threshold (default: 0.5)\n"
      << "  --max_boxes INT     Maximum number of boxes (default: 300)\n"
      << "  --device STRING     Device to use: 'gpu' or 'cpu' (default: gpu)\n"
      << "  --help              Show this help message\n";
}

struct Args {
  std::string modelPath;
  std::string videoPath;
  std::string outputPath = "../output/output.mp4";
  float threshold = rfdetr::DEFAULT_CONFIDENCE_THRESHOLD;
  int maxBoxes = rfdetr::DEFAULT_MAX_NUMBER_BOXES;
  std::string device = "gpu";
};

bool parseArgs(int argc, char **argv, Args &args) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--help" || arg == "-h") {
      return false;
    } else if (arg == "--model" && i + 1 < argc) {
      args.modelPath = argv[++i];
    } else if (arg == "--video" && i + 1 < argc) {
      args.videoPath = argv[++i];
    } else if (arg == "--output" && i + 1 < argc) {
      args.outputPath = argv[++i];
    } else if (arg == "--threshold" && i + 1 < argc) {
      args.threshold = std::stof(argv[++i]);
    } else if (arg == "--max_boxes" && i + 1 < argc) {
      args.maxBoxes = std::stoi(argv[++i]);
    } else if (arg == "--device" && i + 1 < argc) {
      args.device = argv[++i];
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      return false;
    }
  }

  if (args.modelPath.empty()) {
    std::cerr << "Error: --model is required\n";
    return false;
  }
  if (args.videoPath.empty()) {
    std::cerr << "Error: --video is required\n";
    return false;
  }

  return true;
}

static void drawDetections(const cv::Mat &image,
                           const std::vector<rfdetr::Detection> &detections,
                           cv::Mat &output) {
  output = image.clone();

  std::map<int, cv::Scalar> labelColors;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 255);

  for (const auto &det : detections) {
    if (labelColors.find(det.label) == labelColors.end()) {
      labelColors[det.label] = cv::Scalar(dis(gen), dis(gen), dis(gen));
    }
  }

  bool hasMasks = false;
  for (const auto &det : detections) {
    if (!det.mask.empty()) {
      hasMasks = true;
      break;
    }
  }

  if (hasMasks) {
    cv::Mat overlay = output.clone();

    for (const auto &det : detections) {
      if (det.mask.empty())
        continue;

      cv::Scalar color = labelColors[det.label];
      cv::Mat colorMask(det.mask.size(), CV_8UC3);
      colorMask.setTo(color, det.mask);

      cv::addWeighted(overlay, 1.0, colorMask, 0.4, 0.0, overlay);
    }

    cv::addWeighted(output, 0.6, overlay, 0.4, 0.0, output);
  }

  for (const auto &det : detections) {
    const auto &box = det.unnormalizedBox;
    cv::Scalar color = labelColors[det.label];

    cv::rectangle(output, box, color, 4);

    std::string text = std::to_string(det.label);
    int baseline = 0;
    cv::Size textSize =
        cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline);

    cv::Point textOrg(box.x + 5, box.y + textSize.height + 5);
    cv::putText(output, text, textOrg, cv::FONT_HERSHEY_SIMPLEX, 0.7, color,
                2);
  }
}

int main(int argc, char **argv) {
  Args args;

  if (!parseArgs(argc, argv, args)) {
    printUsage(argv[0]);
    return args.modelPath.empty() ? 1 : 0;
  }

  try {
    std::cout << "Initializing RF-DETR model..." << std::endl;
    rfdetr::RFDETRModel model(args.modelPath, args.device);

    cv::VideoCapture cap(args.videoPath);
    if (!cap.isOpened()) {
      std::cerr << "Error: Could not open video: " << args.videoPath
                << std::endl;
      return 1;
    }

    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0.0) {
      fps = 30.0;
    }

    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    cv::VideoWriter writer(args.outputPath, fourcc, fps,
                           cv::Size(width, height));
    if (!writer.isOpened()) {
      std::cerr << "Error: Could not open output video: " << args.outputPath
                << std::endl;
      return 1;
    }

    std::cout << "Processing video..." << std::endl;

    cv::Mat frame;
    size_t frameCount = 0;
    while (cap.read(frame)) {
      std::vector<rfdetr::Detection> detections;
      rfdetr::Timings timings;
      model.predict(frame, detections, timings, args.threshold, args.maxBoxes);

      cv::Mat annotated;
      drawDetections(frame, detections, annotated);
      writer.write(annotated);

      ++frameCount;
      if (frameCount % 50 == 0) {
        std::cout << "Processed " << frameCount << " frames..." << std::endl;
      }
    }

    std::cout << "Done! Saved to: " << args.outputPath << std::endl;
    return 0;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
