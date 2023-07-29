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
project(video_read_bitstream)

find_package(holoscan 0.5 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_library(video_read_bitstream SHARED
  video_read_bitstream.hpp
  video_read_bitstream.cpp
  )
add_library(holoscan::ops::video_read_bitstream ALIAS video_read_bitstream)

target_include_directories(video_read_bitstream INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(video_read_bitstream holoscan::core)
