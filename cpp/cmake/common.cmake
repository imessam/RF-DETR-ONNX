cmake_minimum_required(VERSION 3.15...3.31)

# Set C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 -pthread")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_INCLUDE_CURRENT_DIR ON)


# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# !OpenCV

# ONNX Runtime

set(ONNXRUNTIME_ROOT_DIR "/home/imessam/dev/libs/onnx/onnxruntime-linux-x64-gpu-1.21.0")
if(ONNXRUNTIME_ROOT_DIR)
    set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT_DIR}/include")
    find_library(ONNXRUNTIME_LIBRARY
        NAMES onnxruntime
        PATHS "${ONNXRUNTIME_ROOT_DIR}/lib"
        NO_DEFAULT_PATH
    )
    if(ONNXRUNTIME_LIBRARY)
        set(onnxruntime_FOUND TRUE)
        message(STATUS "Using manual ONNX Runtime: ${ONNXRUNTIME_LIBRARY}")
    else()
        message(FATAL_ERROR "ONNX Runtime library not found in ${ONNXRUNTIME_ROOT_DIR}/lib")
    endif()
endif()

if(onnxruntime_FOUND)
    include_directories(${ONNXRUNTIME_INCLUDE_DIR})
endif()
# !ONNX Runtime



include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
