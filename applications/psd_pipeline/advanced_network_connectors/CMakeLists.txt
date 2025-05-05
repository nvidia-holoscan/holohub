# SPDX-FileCopyrightText: 2024 Valley Tech Systems, Inc.
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20)

find_package(holoscan 2.6.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")
find_package(matx REQUIRED CONFIG)

set(CMAKE_CUDA_ARCHITECTURES "70;80;90")
enable_language(CUDA)

add_library(advanced_network_connectors
  vita49_rx.cu
)

target_link_libraries(advanced_network_connectors PRIVATE
  matx::matx
  holoscan::core
  holoscan::advanced_network
)

target_include_directories(advanced_network_connectors PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../operators/advanced_network
)
