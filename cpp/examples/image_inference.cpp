#include "rfdetr_model.hpp"
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

void print_usage(const char *program_name) {
  std::cout
      << "Usage: " << program_name << " [OPTIONS]\n"
      << "Run image inference with a RF-DETR ONNX model.\n\n"
      << "Required arguments:\n"
      << "  --model PATH        Path to the ONNX model file\n"
      << "  --image PATH        Path to the input image\n\n"
      << "Optional arguments:\n"
      << "  --output PATH       Path to save output image (default: "
         "../output/output.jpg)\n"
      << "  --threshold FLOAT   Confidence threshold (default: 0.5)\n"
      << "  --max_boxes INT     Maximum number of boxes (default: 300)\n"
      << "  --device STRING     Device to use: 'gpu' or 'cpu' (default: gpu)\n"
      << "  --help              Show this help message\n";
}

struct Args {
  std::string model_path;
  std::string image_path;
  std::string output_path = "../output/output.jpg";
  float threshold = rfdetr::DEFAULT_CONFIDENCE_THRESHOLD;
  int max_boxes = rfdetr::DEFAULT_MAX_NUMBER_BOXES;
  std::string device = "gpu";
};

bool parse_args(int argc, char **argv, Args &args) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--help" || arg == "-h") {
      return false;
    } else if (arg == "--model" && i + 1 < argc) {
      args.model_path = argv[++i];
    } else if (arg == "--image" && i + 1 < argc) {
      args.image_path = argv[++i];
    } else if (arg == "--output" && i + 1 < argc) {
      args.output_path = argv[++i];
    } else if (arg == "--threshold" && i + 1 < argc) {
      args.threshold = std::stof(argv[++i]);
    } else if (arg == "--max_boxes" && i + 1 < argc) {
      args.max_boxes = std::stoi(argv[++i]);
    } else if (arg == "--device" && i + 1 < argc) {
      args.device = argv[++i];
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      return false;
    }
  }

  if (args.model_path.empty()) {
    std::cerr << "Error: --model is required\n";
    return false;
  }
  if (args.image_path.empty()) {
    std::cerr << "Error: --image is required\n";
    return false;
  }

  return true;
}

int main(int argc, char **argv) {
  Args args;

  if (!parse_args(argc, argv, args)) {
    print_usage(argv[0]);
    return args.model_path.empty() ? 1 : 0;
  }

  try {
    std::cout << "Initializing RF-DETR model..." << std::endl;
    rfdetr::RFDETRModel model(args.model_path, args.device);

    std::cout << "Loading image: " << args.image_path << std::endl;
    cv::Mat image = cv::imread(args.image_path);
    if (image.empty()) {
      std::cerr << "Error: Could not load image: " << args.image_path
                << std::endl;
      return 1;
    }

    std::cout << "Running inference..." << std::endl;
    std::vector<detectiondata::Detection> detections;
    rfdetr::Timings timings;
    model.predict(image, detections, timings, args.threshold, args.max_boxes);

    float processing_time =
        timings.preprocess + timings.ort_run + timings.postprocess;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n--- Inference Results ---\n";
    std::cout << "Preprocessing:  " << timings.preprocess << " ms\n";
    std::cout << "ORT Run:        " << timings.ort_run << " ms\n";
    std::cout << "Postprocessing: " << timings.postprocess << " ms\n";
    std::cout << "---------------------------------\n";
    std::cout << "Processing (Pre+ORT+Post): " << processing_time << " ms\n";
    std::cout << "Processing FPS:           " << (1000.0f / processing_time)
              << "\n";
    std::cout << "---------------------------------\n";
    std::cout << "Total Latency (inc. I/O):  " << timings.total << " ms\n";
    std::cout << "Total FPS:                " << (1000.0f / timings.total)
              << "\n";
    std::cout << "---------------------------------\n";
    std::cout << "Detections found: " << detections.size() << "\n";

    std::cout << "Saving detections to: " << args.output_path << std::endl;
    model.saveDetections(image, detections, args.output_path);

    std::cout << "Done!\n";

    return 0;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
