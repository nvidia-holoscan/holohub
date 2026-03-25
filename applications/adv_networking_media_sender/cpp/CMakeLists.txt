# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(adv_networking_media_sender CXX)

# This application requires the Rivermax ANO manager
if(NOT ANO_MGR STREQUAL "rivermax")
  message(STATUS "adv_networking_media_sender requires ANO_MGR=rivermax - skipping")
  return()
endif()

# Dependencies
find_package(holoscan 2.6 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES "70;80;90")

# Create the executable
add_executable(${PROJECT_NAME}
  adv_networking_media_sender.cpp
)

target_link_libraries(${PROJECT_NAME}
  PRIVATE
  holoscan::core
  holoscan::advanced_network
  holoscan::ops::video_stream_replayer
  holoscan::ops::advanced_network_media_tx
)

# Copy config file to build tree
add_custom_target(adv_networking_media_sender_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/../adv_networking_media_sender.yaml"
                                   "${CMAKE_CURRENT_BINARY_DIR}/adv_networking_media_sender.yaml"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../adv_networking_media_sender.yaml"
)
add_dependencies(${PROJECT_NAME} adv_networking_media_sender_yaml)
