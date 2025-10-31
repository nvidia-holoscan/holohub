# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(nv_video_reader LANGUAGES CXX)

find_package(holoscan 3.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Detect system architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    set(LIB_ARCH_DIR "aarch64-linux-gnu")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
    set(LIB_ARCH_DIR "x86_64-linux-gnu")
else()
    message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# Reader Operator
add_library(nv_video_reader SHARED
  nv_video_reader.cpp
  ${NVC_SDK_PATH}/FFmpegDemuxer.h
  ${NVC_SDK_PATH}/Logger.cpp
)

add_library(holoscan::ops::nv_video_reader ALIAS nv_video_reader)
target_link_libraries(nv_video_reader PUBLIC
  holoscan::core
  avcodec
  avformat
  avutil
)
target_include_directories(nv_video_reader PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${NVC_SDK_PATH}/>
  $<BUILD_INTERFACE:${NVC_SDK_PATH}/Utils>
)

if(HOLOHUB_BUILD_PYTHON)
  add_subdirectory(python)
endif()
