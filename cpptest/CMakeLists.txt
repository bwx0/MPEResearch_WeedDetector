cmake_minimum_required(VERSION 3.10)
project(MyProject)

# Specify the required C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cpp)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Add executable
add_executable(VITest vitest.cpp)
add_executable(HoughTest houghtest.cpp)

# Link libraries
target_link_libraries(VITest ${OpenCV_LIBS})
target_link_libraries(HoughTest ${OpenCV_LIBS})
