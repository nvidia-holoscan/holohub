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

project(point_cloud_from_depth LANGUAGES CXX CUDA)

find_package(holoscan REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_library(point_cloud_from_depth SHARED
  point_cloud_from_depth.cpp
  point_cloud_from_depth.hpp
  deproject.cu
  deproject.hpp
)

set_target_properties(point_cloud_from_depth
  PROPERTIES
    # compile for the architecture of the current GPU
    CUDA_ARCHITECTURES "native"
)

target_link_libraries(point_cloud_from_depth
  PUBLIC
    holoscan::core
)

target_include_directories(point_cloud_from_depth
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(HOLOHUB_BUILD_PYTHON)
    add_subdirectory(python)
endif()

if(BUILD_TESTING)
    add_subdirectory(test)
endif()

install(TARGETS point_cloud_from_depth)
