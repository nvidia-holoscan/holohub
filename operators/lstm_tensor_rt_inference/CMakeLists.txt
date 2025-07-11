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
project(lstm_tensor_rt_inference)

find_package(holoscan REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_library(lstm_tensor_rt_inference SHARED
  lstm_tensor_rt_inference.cpp
  lstm_tensor_rt_inference.hpp
  )

target_link_libraries(lstm_tensor_rt_inference holoscan::core)
target_include_directories(lstm_tensor_rt_inference INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

if(HOLOHUB_BUILD_PYTHON)
    add_subdirectory(python)
endif()

# Installation
install(TARGETS lstm_tensor_rt_inference)

