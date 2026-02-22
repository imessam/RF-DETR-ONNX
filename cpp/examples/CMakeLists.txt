cmake_minimum_required(VERSION 3.15...3.31)

project(rfdetr_examples)

include(${CMAKE_CURRENT_LIST_DIR}/../cmake/common.cmake)

add_executable(rfdetr_image_inference image_inference.cpp)
target_link_libraries(rfdetr_image_inference PRIVATE rfdetr_onnx)
target_include_directories(rfdetr_image_inference PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../include)

add_executable(rfdetr_video_inference video_inference.cpp)
target_link_libraries(rfdetr_video_inference PRIVATE rfdetr_onnx)
target_include_directories(rfdetr_video_inference PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../include)
