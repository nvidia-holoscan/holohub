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
project(prohawk_video_processing)

find_package(holoscan 0.5 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Finds the Prohawk SDK
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})
find_package(Prohawk REQUIRED)
find_package(OpenCV REQUIRED)

add_library(prohawk_video_processing SHARED
  prohawkop.cpp
  prohawkop.hpp
  )
add_library(holoscan::prohawk_video_processing ALIAS prohawk_video_processing)

target_link_libraries(prohawk_video_processing holoscan::core Prohawk::ptgde ${OpenCV_LIBS})
target_include_directories(prohawk_video_processing INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

if(HOLOHUB_BUILD_PYTHON)
    add_subdirectory(python)
endif()
