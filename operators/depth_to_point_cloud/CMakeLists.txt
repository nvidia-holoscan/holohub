# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

project(depth_to_point_cloud LANGUAGES CXX CUDA)

find_package(holoscan 4.0.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_library(depth_to_point_cloud SHARED
  depth_to_point_cloud.cpp
  depth_to_point_cloud.hpp
  deproject.cu
  deproject.hpp
)

set_target_properties(depth_to_point_cloud
  PROPERTIES
    # compile for the architecture of the current GPU
    CUDA_ARCHITECTURES "native"
)

target_link_libraries(depth_to_point_cloud
  PUBLIC
    holoscan::core
)

target_include_directories(depth_to_point_cloud
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(HOLOHUB_BUILD_PYTHON)
    add_subdirectory(python)
endif()

if(BUILD_TESTING)
    add_subdirectory(test)
endif()

install(TARGETS depth_to_point_cloud)
