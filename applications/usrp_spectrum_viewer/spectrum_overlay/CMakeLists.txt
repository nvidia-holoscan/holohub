# SPDX-FileCopyrightText: 2026 National Instruments Corporation
#
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20)

find_package(holoscan 4.4.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_library(spectrum_overlay
  spectrum_overlay.cpp
)
target_link_libraries(spectrum_overlay
  PRIVATE
  holoscan::core
  holoscan::ops::holoviz
  holoscan::viz::imgui
)
target_include_directories(spectrum_overlay
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/..
)
