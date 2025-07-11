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

cmake_minimum_required(VERSION 3.24)
project(velodyne_lidar
  LANGUAGES CXX
)

# Enable cuda language
set(CMAKE_CUDA_ARCHITECTURES "70;80")
enable_language(CUDA)

find_package(holoscan 2.0 REQUIRED CONFIG)

add_library(velodyne_lidar SHARED
  velodyne_convert_xyz.cu
  velodyne_lidar.cpp
)

target_link_libraries(velodyne_lidar
  holoscan::core
  basic_network
)
target_include_directories(velodyne_lidar INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
