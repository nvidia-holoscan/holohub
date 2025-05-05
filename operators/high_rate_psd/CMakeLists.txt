# SPDX-FileCopyrightText: 2024 Valley Tech Systems, Inc.
#
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20)
project(high_rate_psd CXX)

set(CMAKE_CUDA_ARCHITECTURES "70;80;90")
enable_language(CUDA)

find_package(holoscan 2.5.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")
find_package(matx CONFIG REQUIRED)

add_library(high_rate_psd
  high_rate_psd.cu
  high_rate_psd.hpp
)
add_library(holoscan::ops::high_rate_psd ALIAS high_rate_psd)
target_include_directories(high_rate_psd INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(high_rate_psd
  PRIVATE
    holoscan::core
    matx::matx
)

install(TARGETS high_rate_psd)

