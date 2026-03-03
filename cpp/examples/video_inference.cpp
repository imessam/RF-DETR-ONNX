#define STRING(x) #x
#define XSTRING(x) STRING(x)

#include "rfdetr_model.hpp"
#include "utils.hpp"
#include <iostream>
#include <string>

void print_usage(const char *program_name) {
  std::cout
      << "Usage: " << program_name << " [OPTIONS]\n"
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
  std::string _source_root = XSTRING(SOURCE_ROOT_DETECTION);
  std::string model_path;
  std::string video_path;
  std::string output_path = _source_root + "/output/output.mp4";
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
    } else if (arg == "--video" && i + 1 < argc) {
      args.video_path = argv[++i];
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
  if (args.video_path.empty()) {
    std::cerr << "Error: --video is required\n";
    return false;
  }

  return true;
}

int main(int argc, char **argv) {
  Args args;

  if (!parse_args(argc, argv, args)) {
    print_usage(argv[0]);
    return args.model_path.empty() && args.video_path.empty() ? 1 : 0;
  }

  try {
    std::cout << "Initializing RF-DETR model..." << std::endl;
    rfdetr::RFDETRModel model(args.model_path, args.device);

    cv::VideoCapture cap(args.video_path);
    if (!cap.isOpened()) {
      std::cerr << "Error: Could not open video: " << args.video_path
                << std::endl;
      return 1;
    }

    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0.0) {
      fps = 30.0;
    }

    int fourcc = cv::VideoWriter::fourcc('a', 'v', 'c', '1');
    cv::VideoWriter writer(args.output_path, fourcc, fps,
                           cv::Size(width, height));
    if (!writer.isOpened()) {
      std::cerr << "Error: Could not open output video: " << args.output_path
                << std::endl;
      return 1;
    }

    std::cout << "Processing video..." << std::endl;

    cv::Mat frame;
    size_t frame_count = 0;
    while (cap.read(frame)) {
      std::vector<rfdetr::Detection> detections;
      rfdetr::Timings timings;
      model.predict(frame, detections, timings, args.threshold, args.max_boxes);

      cv::Mat annotated;
      double current_fps = 1000.0 / timings.total;
      rfdetr::drawDetections(frame, detections, annotated, current_fps);
      writer.write(annotated);

      ++frame_count;
      if (frame_count % 50 == 0) {
        std::cout << "Processed " << frame_count << " frames..." << std::endl;
        std::cout << "  - PRE:   " << timings.preprocess << " ms" << std::endl;
        std::cout << "  - ORT:   " << timings.ort_run << " ms" << std::endl;
        std::cout << "  - POST:  " << timings.postprocess << " ms" << std::endl;
        std::cout << "  - TOTAL: " << timings.total << " ms ("
                  << (1000.0 / timings.total) << " FPS)" << std::endl;
      }
      if (frame_count == 1000) {
        break;
      }
    }

    std::cout << "Done! Saved to: " << args.output_path << std::endl;
    return 0;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
