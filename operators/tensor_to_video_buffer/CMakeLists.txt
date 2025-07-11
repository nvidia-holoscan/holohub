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
cmake_minimum_required(VERSION 3.20)
project(tensor_to_video_buffer)

find_package(holoscan 0.5 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_library(tensor_to_video_buffer SHARED
  tensor_to_video_buffer.hpp
  tensor_to_video_buffer.cpp
  )
add_library(holoscan::ops::tensor_to_video_buffer ALIAS tensor_to_video_buffer)

target_include_directories(tensor_to_video_buffer INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(
  tensor_to_video_buffer
  holoscan::core
)

if(HOLOHUB_BUILD_PYTHON)
    add_subdirectory(python)
endif()
