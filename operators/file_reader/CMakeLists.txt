# SPDX-FileCopyrightText: 2024 Valley Tech Systems, Inc.
#
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20)
project(file_reader CXX)

set(CMAKE_CUDA_ARCHITECTURES "70;80;90")
enable_language(CUDA)

find_package(holoscan 2.5.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

include(FetchContent)
FetchContent_Declare(
  MatX
  GIT_REPOSITORY https://github.com/NVIDIA/MatX.git
  GIT_TAG v0.9.0
)
FetchContent_MakeAvailable(MatX)

add_library(vrt_time
  vrt_time.cpp
  vrt_time.hpp
)

add_library(file_reader
  file_reader.cu
  file_reader.hpp
)
add_library(holoscan::ops::file_reader ALIAS file_reader)
target_include_directories(file_reader INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(file_reader
  PRIVATE
    holoscan::core
    matx::matx
    vrt_time
)

install(TARGETS file_reader)
