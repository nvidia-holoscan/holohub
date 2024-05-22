# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.20.1)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
find_package(holoscan 2.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")
project(xr_basic_render_op)

add_library(xr_basic_render_op
  xr_basic_render_op.cpp
  xr_basic_render_op.hpp
)

target_include_directories(xr_basic_render_op
 PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(xr_basic_render_op
  holoscan::core
  holoscan::ops::frame_op
  holoscan::ops::xr_transform_op
  holoscan::viz
)

add_library(holoscan::ops::xr_basic_render_op ALIAS xr_basic_render_op)
