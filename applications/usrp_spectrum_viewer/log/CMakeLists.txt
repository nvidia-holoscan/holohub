# SPDX-FileCopyrightText: 2026 National Instruments Corporation
#
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20)

find_package(holoscan 4.4.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

enable_language(CUDA)

add_library(log
  log.cu
)
target_link_libraries(log
  PRIVATE
  holoscan::matx
  holoscan::core
)
target_include_directories(log
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/..
)
