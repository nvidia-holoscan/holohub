# SPDX-FileCopyrightText: 2026 National Instruments Corporation
#
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20)

# Dependencies
find_package(holoscan 4.4.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

if(DEFINED ENV{DAQIRI_ROOT})
  list(PREPEND CMAKE_PREFIX_PATH "$ENV{DAQIRI_ROOT}")
endif()
list(PREPEND CMAKE_PREFIX_PATH "/opt/daqiri")
find_package(daqiri REQUIRED CONFIG)
include(HoloHubDaqiri)

enable_language(CUDA)

# Create the library
add_library(uhd_chdr_rx
  uhd_chdr_rx.cu
)
target_link_libraries(uhd_chdr_rx PRIVATE
  holoscan::matx
  holoscan::core
)
holohub_link_daqiri(uhd_chdr_rx)
