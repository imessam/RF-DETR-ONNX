cmake_minimum_required(VERSION 3.15...3.31)

project(RFDETRONNXTest VERSION 1.0
                  DESCRIPTION "RF-DETR-ONNX Test"
                  LANGUAGES CXX)

# Top-level paths
set(REPO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
set(CPP_ROOT ${REPO_ROOT}/cpp)

include(${CMAKE_CURRENT_LIST_DIR}/../cmake/common.cmake)

# Link against the library we already built
link_directories(${CPP_ROOT}/build/src)

add_executable(
  generate_onnx
  ${CMAKE_CURRENT_LIST_DIR}/generate_onnx.cpp
)

target_include_directories(generate_onnx PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../include)

target_link_libraries(
  generate_onnx
  rfdetr_onnx
  ${OpenCV_LIBS}
  ${ONNXRUNTIME_LIBRARY}
)

add_executable(
  test_utils
  ${CMAKE_CURRENT_LIST_DIR}/test_utils.cpp
)

target_include_directories(test_utils PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../include)

target_link_libraries(
  test_utils
  rfdetr_onnx
  ${OpenCV_LIBS}
  gtest
  gtest_main
)

add_executable(
  test_model
  ${CMAKE_CURRENT_LIST_DIR}/test_model.cpp
)

target_include_directories(test_model PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../include)

target_link_libraries(
  test_model
  rfdetr_onnx
  ${OpenCV_LIBS}
  gtest
  gtest_main
)


