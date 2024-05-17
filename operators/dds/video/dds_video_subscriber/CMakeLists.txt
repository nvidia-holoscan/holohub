# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(dds_video_subscriber)

find_package(holoscan 2.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Build the dds operator base
set("BUILD_dds_operator_base" ON CACHE BOOL "Build dds_operator_base" FORCE)

# Subscriber Operator
add_library(dds_video_subscriber SHARED dds_video_subscriber.cpp)
add_library(holoscan::ops::dds_video_subscriber ALIAS dds_video_subscriber)
target_link_libraries(dds_video_subscriber PUBLIC
  dds_operator_base
  dds_video_frame
)
target_include_directories(dds_video_subscriber PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

if(HOLOHUB_BUILD_PYTHON)
  add_subdirectory(python)
endif()
