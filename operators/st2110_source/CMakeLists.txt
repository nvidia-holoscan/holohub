# SPDX-FileCopyrightText: Copyright (c) 2025-2026, XRlabs. All rights reserved.
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

project(st2110_source LANGUAGES CXX CUDA)

find_package(holoscan REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_library(st2110_source SHARED
  st2110_source.cpp
  st2110_source.hpp
  st2110_kernels.cu
  st2110_kernels.cuh
)

add_library(holoscan::ops::st2110_source ALIAS st2110_source)

set_target_properties(st2110_source
  PROPERTIES
    # Compile for the architecture of the current GPU
    CUDA_ARCHITECTURES "native"
)

target_link_libraries(st2110_source
  PUBLIC
    holoscan::core
    CUDA::cuda_driver
)

target_include_directories(st2110_source
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(HOLOHUB_BUILD_PYTHON)
    add_subdirectory(python)
endif()

install(TARGETS st2110_source)
