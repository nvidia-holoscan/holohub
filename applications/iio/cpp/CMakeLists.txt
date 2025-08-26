# SPDX-FileCopyrightText: Copyright (c) 2025 Analog Devices, Inc. All rights reserved.
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
project(iio LANGUAGES CXX)

set(CMAKE_CUDA_ARCHITECTURES "70;80;89;90")
enable_language(CUDA)

find_package(
  holoscan
  2.6
  REQUIRED
  CONFIG
  PATHS
  "/opt/nvidia/holoscan"
  "/workspace/holoscan-sdk/install")

find_package(matx CONFIG REQUIRED)

add_executable(iio main.cpp pluto_fft_example/pluto_fft_example.hpp
                   pluto_fft_example/support_operators.hpp)

target_link_libraries(
  iio PRIVATE holoscan::core iio_controller holoscan::ops::fft matx::matx
              iio_basic_example)

# Copy config file
add_custom_target(
  iio_config_yaml
  COMMAND
    ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/../iio_config.yaml"
    ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../iio_config.yaml"
  BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/iio_config.yaml")

add_dependencies(iio iio_config_yaml)

# Add subdirectories
add_subdirectory(pluto_fft_example)
add_subdirectory(basic_example)
