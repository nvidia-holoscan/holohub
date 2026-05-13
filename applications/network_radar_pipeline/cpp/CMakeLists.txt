# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
cmake_minimum_required(VERSION 3.20)
project(network_radar_pipeline CXX CUDA)

find_package(holoscan 0.6 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")
if(DEFINED ENV{DAQIRI_ROOT})
  list(PREPEND CMAKE_PREFIX_PATH "$ENV{DAQIRI_ROOT}")
endif()
list(PREPEND CMAKE_PREFIX_PATH "/opt/daqiri")
find_package(daqiri REQUIRED CONFIG)
include(HoloHubDaqiri)

# set(CMAKE_BUILD_TYPE Debug)
# Enable cuda language
set(CMAKE_CUDA_ARCHITECTURES "70;80;90")
enable_language(CUDA)

# Main
add_executable(network_radar_pipeline
  main.cpp
  source.cu
  process.cu
)

# set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -ggdb")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -ggdb")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")

add_subdirectory(daqiri_connectors)

target_link_libraries(network_radar_pipeline PRIVATE
  holoscan::matx
  holoscan::core
  daqiri_connectors
)
holohub_link_daqiri(network_radar_pipeline)

set_target_properties(network_radar_pipeline PROPERTIES
  CUDA_ARCHITECTURES "80;90"
)

target_include_directories(network_radar_pipeline PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# Copy config file
add_custom_target(source_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/source.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source.yaml"
)
add_custom_target(process_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/process.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/process.yaml"
)

add_dependencies(network_radar_pipeline source_yaml process_yaml)
