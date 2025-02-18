# SPDX-FileCopyrightText: 2024 Valley Tech Systems, Inc.
#
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20)
project(psd_pipeline CXX CUDA)

find_package(holoscan 2.5.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

set(CMAKE_CUDA_ARCHITECTURES "70;80;90")
enable_language(CUDA)

add_subdirectory(advanced_network_connectors)
add_subdirectory(data_writer)

add_executable(psd_pipeline
  main.cpp
)

# Download MatX
include(FetchContent)
FetchContent_Declare(
  MatX
  GIT_REPOSITORY https://github.com/NVIDIA/MatX.git
  GIT_TAG v0.9.0
)
FetchContent_MakeAvailable(MatX)

target_link_libraries(psd_pipeline PRIVATE
  matx::matx
  holoscan::core
  holoscan::advanced_network_rx
  holoscan::ops::fft
  holoscan::ops::high_rate_psd
  holoscan::ops::low_rate_psd
  holoscan::ops::vita49_psd_packetizer
  holoscan::ops::data_writer
  advanced_network_connectors
)
target_include_directories(psd_pipeline PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

add_custom_target(config_yaml
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/config.yaml ${CMAKE_CURRENT_BINARY_DIR}/config.yaml
)

add_dependencies(psd_pipeline config_yaml)
