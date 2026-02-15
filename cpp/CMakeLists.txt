cmake_minimum_required(VERSION 3.15...3.31)

project(RFDETRONNXDemo VERSION 1.0
                  DESCRIPTION "RF-DETR-ONNX Demo"
                  LANGUAGES CXX)


option(ENABLE_TESTS "isEnableTests" OFF)



set(EXECS)
set(LIBS)
set(TESTS)

## Libraries setup

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/src)

set(LIBS rfdetr_onnx)



## Executables setup

add_executable(rfdetr_onnx_demo ${CMAKE_CURRENT_LIST_DIR}/demo.cpp)

set(EXECS ${EXECS} rfdetr_onnx_demo)

foreach(_target
${EXECS})
  target_link_libraries(${_target}
    ${LIBS}
    )
endforeach()


## Tests setup
if(${ENABLE_TESTS})

  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/test)

#   include(${CMAKE_CURRENT_LIST_DIR}/../test-libs/another_module/cpp/src/CMakeLists.txt)
  # include(${CMAKE_CURRENT_LIST_DIR}/../test-libs/another_module/libs/CMakeLists.txt)

#   set(test_libs AnotherModuleLib ${libs_targets_another_module})

  set(TESTS rfdetr_onnx_test)



  foreach(_test
  ${TESTS})
    target_link_libraries(${_test}
        ${LIBS}
      )
  endforeach()

endif()
