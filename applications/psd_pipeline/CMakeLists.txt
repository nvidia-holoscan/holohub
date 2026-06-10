# SPDX-FileCopyrightText: 2024-2026 Voyager Technologies, Inc.
#
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.24)
project(psd_pipeline CXX CUDA)

find_package(holoscan 4.0.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")
find_package(CUDAToolkit REQUIRED)
if(DEFINED ENV{DAQIRI_ROOT})
  list(PREPEND CMAKE_PREFIX_PATH "$ENV{DAQIRI_ROOT}")
endif()
list(PREPEND CMAKE_PREFIX_PATH "/opt/daqiri")
find_package(daqiri REQUIRED CONFIG)
include(HoloHubDaqiri)

set(CMAKE_CUDA_ARCHITECTURES "70;80;90")
enable_language(CUDA)

add_subdirectory(daqiri_connectors)
add_subdirectory(data_writer)

add_executable(psd_pipeline
  main.cpp
)

target_link_libraries(psd_pipeline PRIVATE
  holoscan::core
  holoscan::ops::fft
  holoscan::ops::high_rate_psd
  holoscan::ops::low_rate_psd
  holoscan::ops::vita49_psd_packetizer
  holoscan::ops::data_writer
  holoscan::matx
  CUDA::cufft
  CUDA::cublas
  CUDA::cusolver
  daqiri_connectors
)
holohub_link_daqiri(psd_pipeline)

add_custom_target(config_yaml
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/config.yaml ${CMAKE_CURRENT_BINARY_DIR}/config.yaml
)

add_dependencies(psd_pipeline config_yaml)
