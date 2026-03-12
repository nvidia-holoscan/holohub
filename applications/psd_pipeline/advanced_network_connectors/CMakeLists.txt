# SPDX-FileCopyrightText: 2024, 2026 Valley Tech Systems, Inc.
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20)

find_package(holoscan 2.6.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

set(CMAKE_CUDA_ARCHITECTURES "70;80;90")
enable_language(CUDA)

add_library(advanced_network_connectors
  vita49_rx.cu
)

target_link_libraries(advanced_network_connectors PRIVATE
  holoscan::core
  holoscan::advanced_network
  holoscan::matx
)