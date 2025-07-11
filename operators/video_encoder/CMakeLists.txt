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
project(video_encoder CXX)

find_package(holoscan 2.1 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")


add_library(video_encoder_request SHARED
  video_encoder_request/video_encoder_request.hpp
)
set_target_properties(video_encoder_request PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(video_encoder_request PUBLIC holoscan::core)

add_library(video_encoder INTERFACE)
target_link_libraries(video_encoder INTERFACE video_encoder_request
)

add_library(holoscan::ops::video_encoder ALIAS video_encoder)

target_include_directories(video_encoder INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
