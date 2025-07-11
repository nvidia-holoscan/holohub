# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

project(tool_tracking_postprocessor LANGUAGES CXX CUDA)

find_package(holoscan REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_library(tool_tracking_postprocessor SHARED
  tool_tracking_postprocessor.cpp
  tool_tracking_postprocessor.hpp
  tool_tracking_postprocessor.cu
  tool_tracking_postprocessor.cuh
  )

set_target_properties(tool_tracking_postprocessor
  PROPERTIES
    # separable compilation is required since we launch kernels from within kernels
    CUDA_SEPARABLE_COMPILATION ON
    # compile for the architecture of the current GPU
    CUDA_ARCHITECTURES "native"
  )

target_link_libraries(tool_tracking_postprocessor
  PUBLIC
    holoscan::core
  )

target_include_directories(tool_tracking_postprocessor
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )

if(HOLOHUB_BUILD_PYTHON)
    add_subdirectory(python)
endif()

install(TARGETS tool_tracking_postprocessor)

