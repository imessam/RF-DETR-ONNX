cmake_minimum_required(VERSION 3.10)
project(rfdetr_benchmarks)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(OpenCV REQUIRED)

# Top-level paths
set(REPO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
set(CPP_ROOT ${REPO_ROOT}/cpp)

# Include directories
include_directories(
    ${CPP_ROOT}/include
    ${OpenCV_INCLUDE_DIRS}
    /home/imessam/dev/libs/onnx/onnxruntime-linux-x64-gpu-1.21.0/include
)

# ONNX Runtime manually
set(ONNXRUNTIME_LIB /home/imessam/dev/libs/onnx/onnxruntime-linux-x64-gpu-1.21.0/lib/libonnxruntime.so)

# Link against the library we already built
link_directories(${CPP_ROOT}/build/src)

add_executable(rfdetr_benchmark benchmark.cpp)

target_link_libraries(rfdetr_benchmark
    PRIVATE
    rfdetr_onnx
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIB}
)

# Ensure it finds the header files
target_include_directories(rfdetr_benchmark PRIVATE ${CPP_ROOT}/include)
