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
cmake_minimum_required(VERSION 3.20)
project(video_decoder)

find_package(holoscan 0.6 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_library(video_decoder_context SHARED
  video_decoder_context/video_decoder_context.hpp
  video_decoder_context/video_decoder_context.cpp
)
target_link_libraries(video_decoder_context PUBLIC holoscan::core)

add_library(video_decoder_request SHARED
  video_decoder_request/video_decoder_request.hpp
  video_decoder_request/video_decoder_request.cpp
)
target_link_libraries(video_decoder_request PUBLIC holoscan::core PRIVATE video_decoder_context)

add_library(video_decoder_response SHARED
  video_decoder_response/video_decoder_response.hpp
  video_decoder_response/video_decoder_response.cpp
)
target_link_libraries(video_decoder_response PUBLIC holoscan::core PRIVATE video_decoder_context)

add_library(video_decoder INTERFACE)
target_link_libraries(video_decoder
INTERFACE
  video_decoder_context
  video_decoder_request
  video_decoder_response
)

add_library(holoscan::ops::video_decoder ALIAS video_decoder)

target_include_directories(video_decoder INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})


add_subdirectory(video_decoder_context)
add_subdirectory(video_decoder_request)
add_subdirectory(video_decoder_response)
