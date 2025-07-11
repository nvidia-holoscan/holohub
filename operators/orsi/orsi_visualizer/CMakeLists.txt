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
project(orsi_visualizer)

add_subdirectory(lib)

find_package(holoscan 0.5 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# -------------------------------------------------------------------------------
#
# Holoscan operator
#

add_library(orsi_visualizer SHARED
  orsi_visualizer.hpp
  orsi_visualizer.cpp
)
add_library(holoscan::orsi::visualizer ALIAS orsi_visualizer)
target_include_directories(orsi_visualizer INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(orsi_visualizer
  holoscan::core
  orsi_visualizer_lib
)

if(HOLOHUB_BUILD_PYTHON)
  add_subdirectory(python)
endif()
