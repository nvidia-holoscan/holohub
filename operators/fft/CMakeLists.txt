# SPDX-FileCopyrightText: 2024-2026 Voyager Technologies, Inc.
#
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20)
project(fft CXX)

set(CMAKE_CUDA_ARCHITECTURES "70;80;90")
enable_language(CUDA)

find_package(holoscan 4.0.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_library(fft
  fft.cu
  fft.hpp
)
add_library(holoscan::ops::fft ALIAS fft)
target_include_directories(fft INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(fft
  PRIVATE
    holoscan::core
    holoscan::matx
)

install(TARGETS fft)
