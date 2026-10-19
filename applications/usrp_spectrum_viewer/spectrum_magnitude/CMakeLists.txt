# SPDX-FileCopyrightText: 2026 National Instruments Corporation
#
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20)

# Dependencies
find_package(holoscan 4.4.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

enable_language(CUDA)

# Create the library
add_library(spectrum_magnitude
  spectrum_magnitude.cu
)
target_link_libraries(spectrum_magnitude
  PRIVATE
  holoscan::matx
  holoscan::core
)
target_include_directories(spectrum_magnitude
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/..
)
