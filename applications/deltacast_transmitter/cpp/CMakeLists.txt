# SPDX-FileCopyrightText: Copyright (c) 2022-2025 DELTACAST.TV. All rights reserved.
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
project(deltacast_transmitter CXX)

find_package(holoscan 1.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# CPP Applications
add_executable(deltacast_transmitter
  main.cpp
)

target_link_libraries(deltacast_transmitter
  PRIVATE
  holoscan::core
  holoscan::ops::video_stream_replayer
  holoscan::ops::format_converter
  holoscan::videomaster
)

# Copy config file
add_custom_target(deltacast_transmitter_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/deltacast_transmitter.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "deltacast_transmitter.yaml"
  BYPRODUCTS "deltacast_transmitter.yaml"
)
add_dependencies(deltacast_transmitter deltacast_transmitter_yaml)
