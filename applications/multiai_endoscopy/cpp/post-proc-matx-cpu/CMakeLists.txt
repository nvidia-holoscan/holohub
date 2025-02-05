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
project(multi_ai CXX)

# Holoscan
find_package(holoscan 0.6 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Enable cuda language
set(CMAKE_CUDA_ARCHITECTURES "70;80")
enable_language(CUDA)

# Uncomment below for debug build
# set(CMAKE_BUILD_TYPE Debug)

# Download MatX
# OBS: Currently pulls a specific commit of MatX (with host support for find_idx)
include(FetchContent)
FetchContent_Declare(
  MatX
  GIT_REPOSITORY https://github.com/NVIDIA/MatX.git
  GIT_TAG main
)
FetchContent_MakeAvailable(MatX)

add_executable(multi_ai
  multi_ai.cpp
)

target_link_libraries(multi_ai
  PRIVATE
  holoscan::core
  holoscan::ops::video_stream_replayer
  holoscan::ops::format_converter
  holoscan::ops::inference
  holoscan::ops::segmentation_postprocessor
  holoscan::ops::holoviz
  holoscan::aja
  matx::matx
)

# Copy config file
add_custom_target(app_config_yaml
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/app_config.yaml" ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS "app_config.yaml"
    BYPRODUCTS "app_config.yaml"
)
add_dependencies(multi_ai app_config_yaml)
