cmake_minimum_required(VERSION 3.15...3.31)

project(RFDETRONNXTest VERSION 1.0
                  DESCRIPTION "RF-DETR-ONNX Test"
                  LANGUAGES CXX)

include(${CMAKE_CURRENT_LIST_DIR}/../cmake/common.cmake)

enable_testing()

add_executable(
  rfdetr_onnx_test

  ${CMAKE_CURRENT_LIST_DIR}/test_main.cpp

  ${CMAKE_CURRENT_LIST_DIR}/test_class_implemented_derived.cpp
)

target_include_directories(rfdetr_onnx_test PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../include)
target_include_directories(rfdetr_onnx_test PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../test-libs/another_module/cpp/include)


target_link_libraries(
  rfdetr_onnx_test
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(rfdetr_onnx_test)
