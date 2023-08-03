# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(network_radar_pipeline CXX)

find_package(holoscan 0.5 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Enable cuda language
set(CMAKE_CUDA_ARCHITECTURES "70;80")
enable_language(CUDA)

# Download MatX
include(FetchContent)
FetchContent_Declare(
  MatX
  GIT_REPOSITORY https://github.com/NVIDIA/MatX.git
  GIT_TAG v0.3.0
)
FetchContent_MakeAvailable(MatX)

# Main
add_executable(network_radar_pipeline
  main.cpp
  source.cu
  process.cu
)

add_subdirectory(advanced_network_connectors)
add_subdirectory(basic_network_connectors)

target_link_libraries(network_radar_pipeline PRIVATE
  matx::matx
  holoscan::core
  holoscan::basic_network
  holoscan::advanced_network
  advanced_network_connectors
  basic_network_connectors
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