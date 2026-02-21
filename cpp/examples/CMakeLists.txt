cmake_minimum_required(VERSION 3.15...3.31)

project(rfdetr_examples)

include(${CMAKE_CURRENT_LIST_DIR}/../cmake/common.cmake)

add_executable(rfdetr_image_inference image_inference.cpp)
target_link_libraries(rfdetr_image_inference PRIVATE rfdetr_onnx)

add_executable(rfdetr_video_inference video_inference.cpp)
target_link_libraries(rfdetr_video_inference PRIVATE rfdetr_onnx)
