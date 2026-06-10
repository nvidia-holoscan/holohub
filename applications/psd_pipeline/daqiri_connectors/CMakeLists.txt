# SPDX-FileCopyrightText: 2024-2026 Voyager Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.24)

find_package(holoscan 4.0.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")
if(DEFINED ENV{DAQIRI_ROOT})
  list(PREPEND CMAKE_PREFIX_PATH "$ENV{DAQIRI_ROOT}")
endif()
list(PREPEND CMAKE_PREFIX_PATH "/opt/daqiri")
find_package(daqiri REQUIRED CONFIG)
include(HoloHubDaqiri)

set(CMAKE_CUDA_ARCHITECTURES "70;80;90")
enable_language(CUDA)

add_library(daqiri_connectors
  vita49_rx.cu
)

target_link_libraries(daqiri_connectors PRIVATE
  holoscan::core
  holoscan::matx
)
holohub_link_daqiri(daqiri_connectors)
