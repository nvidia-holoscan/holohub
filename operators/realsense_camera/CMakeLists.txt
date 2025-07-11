# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(realsense_camera)

find_package(holoscan 2.1 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

find_package(realsense2 REQUIRED)

add_library(realsense_camera SHARED
  realsense_camera.hpp
  realsense_camera.cpp
)
add_library(holoscan::ops::realsense_camera ALIAS realsense_camera)

target_include_directories(
  realsense_camera
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${realsense2_INCLUDE_DIRS}
)

target_link_libraries(
  realsense_camera
  holoscan::core
  ${realsense2_LIBRARY}
)
