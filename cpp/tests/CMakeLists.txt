cmake_minimum_required(VERSION 3.15...3.31)

project(RFDETRONNXTest VERSION 1.0
                  DESCRIPTION "RF-DETR-ONNX Test"
                  LANGUAGES CXX)

include(${CMAKE_CURRENT_LIST_DIR}/../cmake/common.cmake)

add_executable(
  generate_onnx
  ${CMAKE_CURRENT_LIST_DIR}/generate_onnx.cpp
)

target_include_directories(generate_onnx PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../include)

target_link_libraries(
  generate_onnx
  rfdetr_onnx
)

add_executable(
  test_utils
  ${CMAKE_CURRENT_LIST_DIR}/test_utils.cpp
)

target_include_directories(test_utils PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../include)

target_link_libraries(
  test_utils
  rfdetr_onnx
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
  gtest
  gtest_main
)

