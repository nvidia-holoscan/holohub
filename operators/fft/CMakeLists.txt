# SPDX-FileCopyrightText: 2024 Valley Tech Systems, Inc.
#
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20)
project(fft CXX)

set(CMAKE_CUDA_ARCHITECTURES "70;80;90")
enable_language(CUDA)

find_package(holoscan 2.5.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")
find_package(matx CONFIG REQUIRED)

add_library(fft
  fft.cu
  fft.hpp
)
add_library(holoscan::ops::fft ALIAS fft)
target_include_directories(fft INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(fft
  PRIVATE
    holoscan::core
    matx::matx
)

install(TARGETS fft)

