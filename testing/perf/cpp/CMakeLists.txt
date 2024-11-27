cmake_minimum_required(VERSION 3.25)

message("Current CMake directory: ${CMAKE_CURRENT_LIST_DIR}")

set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE} " -std=c++20 -fopenmp -pthread -Ofast -march=native -fprefetch-loop-arrays -g -gdwarf-4 ")

project(perf)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)

include_directories(${pybind11_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

link_directories(${OpenCV_LIB_DIR})


pybind11_add_module(perf_cpp MODULE bindings.cpp hsv.cpp exgi.cpp)
target_link_libraries(perf_cpp PRIVATE ${OpenCV_LIBS})
install(TARGETS perf_cpp DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_executable(exgi_test exgi_test.cpp exgi.cpp)
target_link_libraries(exgi_test PRIVATE ${OpenCV_LIBS})

add_executable(hsv_test hsv_test.cpp hsv.cpp)
target_link_libraries(hsv_test PRIVATE ${OpenCV_LIBS})

add_executable(inplace_test inplace_test.cpp)
target_link_libraries(inplace_test PRIVATE ${OpenCV_LIBS})

