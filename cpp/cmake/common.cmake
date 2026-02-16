cmake_minimum_required(VERSION 3.15...3.31)

# Set C++17 standard (required for ONNX Runtime and modern C++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# # CUDA
# set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
# find_package(CUDA 10.2 REQUIRED)

# # set(CMAKE_CUDA_STANDARD 10.1)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# # !CUDA

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# !OpenCV

# ONNX Runtime
find_package(onnxruntime QUIET)
if(NOT onnxruntime_FOUND)
    # Try to find ONNX Runtime manually
    find_path(ONNXRUNTIME_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        PATHS /usr/include /usr/local/include /home/imessam/dev/libs/onnx/onnxruntime-linux-x64-gpu-1.21.0/include
        PATH_SUFFIXES onnxruntime onnxruntime/core/session
    )
    find_library(ONNXRUNTIME_LIBRARY
        NAMES onnxruntime
        PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu /home/imessam/dev/libs/onnx/onnxruntime-linux-x64-gpu-1.21.0/lib/
    )
    if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
        set(onnxruntime_FOUND TRUE)
        message(STATUS "Found ONNX Runtime: ${ONNXRUNTIME_LIBRARY}")
        message(STATUS "ONNX Runtime include dir: ${ONNXRUNTIME_INCLUDE_DIR}")
        include_directories(${ONNXRUNTIME_INCLUDE_DIR})
    else()
        message(FATAL_ERROR "ONNX Runtime not found. Please install ONNX Runtime.")
    endif()
else()
    # If found via package, ensure include dirs are set
    if(onnxruntime_INCLUDE_DIRS)
        include_directories(${onnxruntime_INCLUDE_DIRS})
    endif()
endif()
# !ONNX Runtime

# find_package(PkgConfig REQUIRED)
# pkg_search_module(gstreamer REQUIRED IMPORTED_TARGET gstreamer-1.0>=1.4)
# pkg_search_module(gstreamer-sdp REQUIRED IMPORTED_TARGET gstreamer-sdp-1.0>=1.4)
# pkg_search_module(gstreamer-app REQUIRED IMPORTED_TARGET gstreamer-app-1.0>=1.4)
# pkg_search_module(gstreamer-video REQUIRED IMPORTED_TARGET gstreamer-video-1.0>=1.4)
# pkg_search_module(gstreamer-pbutils REQUIRED IMPORTED_TARGET gstreamer-pbutils-1.0>=1.4)


include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
