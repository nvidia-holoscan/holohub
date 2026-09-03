# SPDX-FileCopyrightText: 2026 National Instruments Corporation
#
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20)

find_package(holoscan 4.4.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

find_package(UHD 4.10.0.0 REQUIRED)

add_library(usrp_rx
  usrp_rx.cpp
)
target_link_libraries(usrp_rx
  PUBLIC
  holoscan::core
  ${UHD_LIBRARIES}
)
target_include_directories(usrp_rx
  PUBLIC
  ${UHD_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/..
)
