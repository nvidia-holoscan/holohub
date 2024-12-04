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
project(visualizer_icardio LANGUAGES CXX CUDA)

find_package(holoscan 0.5 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_library(visualizer_icardio SHARED
  visualizer_icardio.cpp
  visualizer_icardio.cu
  visualizer_icardio.cuh
  visualizer_icardio.hpp
  )

target_link_libraries(visualizer_icardio
    holoscan::core
    holoscan::infer
    holoscan::infer_utils
    CUDA::cudart)

target_include_directories(visualizer_icardio INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(visualizer_icardio PRIVATE HOLOSCAN_MAJOR_VERSION=${holoscan_VERSION_MAJOR})
target_compile_definitions(visualizer_icardio PRIVATE HOLOSCAN_MINOR_VERSION=${holoscan_VERSION_MINOR})

if(HOLOHUB_BUILD_PYTHON)
    add_subdirectory(python)
endif()
