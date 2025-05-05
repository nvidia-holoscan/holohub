# SPDX-FileCopyrightText: 2024 Valley Tech Systems, Inc.
#
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20)
project(data_writer CXX)

set(CMAKE_CUDA_ARCHITECTURES "70;80;90")
enable_language(CUDA)

find_package(holoscan 2.5.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")
find_package(matx REQUIRED CONFIG)

add_library(data_writer
  data_writer.cu
  data_writer.hpp
)
add_library(holoscan::ops::data_writer ALIAS data_writer)
target_include_directories(data_writer INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(data_writer
  PRIVATE
    holoscan::core
    matx::matx
)

install(TARGETS data_writer)

