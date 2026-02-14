cmake_minimum_required(VERSION 3.15...3.31)

project(RFDETRLib VERSION 1.0
                  DESCRIPTION "RF-DETR-ONNX Library"
                  LANGUAGES CXX)

include(${CMAKE_CURRENT_LIST_DIR}/../cmake/common.cmake)


## Libraries setup

set(RFDETRLib_SOURCES

  ${CMAKE_CURRENT_LIST_DIR}/class_abstract_base.cpp

  ${CMAKE_CURRENT_LIST_DIR}/class_implemented_derived.cpp

)

if (NOT TARGET rfdetr_onnx)

  # include(${CMAKE_CURRENT_LIST_DIR}/../../libs/CMakeLists.txt)

  add_library(rfdetr_onnx STATIC ${RFDETRLib_SOURCES})

  target_include_directories(rfdetr_onnx PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../include)

  foreach(_lib_include_dir
  ${libs_include_rfdetr_onnx})
  target_include_directories(rfdetr_onnx PUBLIC
        ${_lib_include_dir}
      )
  endforeach()

  set(DEPS 

  # yaml-cpp::yaml-cpp
  # PkgConfig::gstreamer
  # PkgConfig::gstreamer-sdp
  # PkgConfig::gstreamer-app
  # PkgConfig::gstreamer-video
  # PkgConfig::gstreamer-pbutils
  # ${OpenCV_LIBS}

)

endif()
