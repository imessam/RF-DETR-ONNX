cmake_minimum_required(VERSION 3.15...3.31)

project(RFDETRONNX VERSION 1.0
                  DESCRIPTION "RF-DETR-ONNX"
                  LANGUAGES CXX)

option(ENABLE_TESTS "Enable C++ tests" OFF)
option(ENABLE_BENCHMARKS "Enable C++ benchmarks" ON)
option(ENABLE_EXAMPLES "Enable C++ examples" ON)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/common.cmake)

include(CMakePackageConfigHelpers)

## Libraries setup
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/src)

## Examples setup
if(ENABLE_EXAMPLES)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/examples)
endif()

## Benchmarks setup
if(ENABLE_BENCHMARKS)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/benchmarks)
endif()

## Tests setup
if(ENABLE_TESTS)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/tests)
endif()

## Install package config
set(RFDETR_ONNX_CMAKE_DIR lib/cmake/rfdetr_onnx)

install(EXPORT rfdetr_onnxTargets
  NAMESPACE rfdetr_onnx::
  DESTINATION ${RFDETR_ONNX_CMAKE_DIR}
)

configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/rfdetr_onnxConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/rfdetr_onnxConfig.cmake
  INSTALL_DESTINATION ${RFDETR_ONNX_CMAKE_DIR}
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/rfdetr_onnxConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/rfdetr_onnxConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/rfdetr_onnxConfigVersion.cmake
  DESTINATION ${RFDETR_ONNX_CMAKE_DIR}
)
