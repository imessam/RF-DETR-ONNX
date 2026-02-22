cmake_minimum_required(VERSION 3.15...3.31)
project(rfdetr_benchmarks)

include(${CMAKE_CURRENT_LIST_DIR}/../cmake/common.cmake)

add_executable(rfdetr_benchmark benchmark.cpp)
target_link_libraries(rfdetr_benchmark PRIVATE rfdetr_onnx)

target_include_directories(rfdetr_benchmark PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../include)

