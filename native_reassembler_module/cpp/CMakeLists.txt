cmake_minimum_required(VERSION 3.25)

message("Current CMake directory: ${CMAKE_CURRENT_LIST_DIR}")

set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE} "-std=c++20 -fopenmp -pthread -Ofast -march=native -fprefetch-loop-arrays -g -gdwarf-4")

project(image_reassembler)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)

include_directories(${pybind11_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

link_directories(${OpenCV_LIB_DIR})


set(ReassemblerSource packer.cpp reassembler.cpp roi_extractor.cpp util.cpp)

# The main module
pybind11_add_module(native_reassembler_module MODULE bindings.cpp ${ReassemblerSource})
target_link_libraries(native_reassembler_module PRIVATE ${OpenCV_LIBS})
install(TARGETS native_reassembler_module DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/..)

# some testing executables
add_executable(reassembler_test ${ReassemblerSource} reassembler_test.cpp)
target_link_libraries(reassembler_test PRIVATE ${OpenCV_LIBS})

add_executable(exgi_test test/exgi_comparison.cpp)
target_link_libraries(exgi_test PRIVATE ${OpenCV_LIBS})

add_executable(exgi_test2 test/exgi_cmp2.cpp util.cpp)
target_link_libraries(exgi_test2 PRIVATE ${OpenCV_LIBS})

add_executable(exgi_video_test test/exgi_vid.cpp ${ReassemblerSource})
target_link_libraries(exgi_video_test PRIVATE ${OpenCV_LIBS})
