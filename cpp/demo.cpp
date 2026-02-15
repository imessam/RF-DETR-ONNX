#include "rfdetr_model.hpp"
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

void printUsage(const char *programName) {
  std::cout
      << "Usage: " << programName << " [OPTIONS]\n"
      << "Run inference with a RF-DETR ONNX model.\n\n"
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
  std::string modelPath;
  std::string imagePath;
  std::string outputPath = "../output/output.jpg";
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
    } else if (arg == "--image" && i + 1 < argc) {
      args.imagePath = argv[++i];
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

  // Validate required arguments
  if (args.modelPath.empty()) {
    std::cerr << "Error: --model is required\n";
    return false;
  }
  if (args.imagePath.empty()) {
    std::cerr << "Error: --image is required\n";
    return false;
  }

  return true;
}

int main(int argc, char **argv) {
  Args args;

  if (!parseArgs(argc, argv, args)) {
    printUsage(argv[0]);
    return args.modelPath.empty() ? 1 : 0; // Return 0 for --help
  }

  try {
    // Initialize the model
    std::cout << "Initializing RF-DETR model..." << std::endl;
    rfdetr::RFDETRModel model(args.modelPath, args.device);

    // Load image
    std::cout << "Loading image: " << args.imagePath << std::endl;
    cv::Mat image = cv::imread(args.imagePath);
    if (image.empty()) {
      std::cerr << "Error: Could not load image: " << args.imagePath
                << std::endl;
      return 1;
    }

    // Run inference
    std::cout << "Running inference..." << std::endl;
    rfdetr::Detection detection;
    rfdetr::Timings timings;
    model.predict(image, detection, timings, args.threshold, args.maxBoxes);

    // Calculate processing time
    float processingTime =
        timings.preprocess + timings.ortRun + timings.postprocess;

    // Print results
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n--- Inference Results ---\n";
    std::cout << "Preprocessing:  " << timings.preprocess << " ms\n";
    std::cout << "ORT Run:        " << timings.ortRun << " ms\n";
    std::cout << "Postprocessing: " << timings.postprocess << " ms\n";
    std::cout << "---------------------------------\n";
    std::cout << "Processing (Pre+ORT+Post): " << processingTime << " ms\n";
    std::cout << "Processing FPS:           " << (1000.0f / processingTime)
              << "\n";
    std::cout << "---------------------------------\n";
    std::cout << "Total Latency (inc. I/O):  " << timings.total << " ms\n";
    std::cout << "Total FPS:                " << (1000.0f / timings.total)
              << "\n";
    std::cout << "---------------------------------\n";
    std::cout << "Detections found: " << detection.boxes.size() << "\n";

    // Draw and save detections
    std::cout << "Saving detections to: " << args.outputPath << std::endl;
    model.saveDetections(image, detection, args.outputPath);
    std::cout << "Done!\n";

    return 0;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}