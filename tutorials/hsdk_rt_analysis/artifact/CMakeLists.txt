cmake_minimum_required(VERSION 3.20.1) # rapids-cmake-src requires >= 3.20.

project(my_project CXX)

find_package(holoscan REQUIRED CONFIG PATHS "/opt/nvidia/holoscan")

add_executable(run_exp experiment.cpp)

target_link_libraries(run_exp
  PRIVATE
    holoscan::core
)

