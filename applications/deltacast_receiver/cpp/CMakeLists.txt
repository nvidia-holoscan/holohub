# SPDX-FileCopyrightText: Copyright (c) 2022-2026, DELTACAST.TV. All rights reserved.
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
project(deltacast_receiver CXX)

find_package(holoscan 1.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# CPP Application
add_executable(deltacast_receiver
  main.cpp
)

target_link_libraries(deltacast_receiver
  PRIVATE
  holoscan::core
  holoscan::ops::format_converter
  holoscan::ops::holoviz
  holoscan::videomaster
)

# Copy config file
add_custom_target(deltacast_receiver_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/deltacast_receiver.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "deltacast_receiver.yaml"
  BYPRODUCTS "deltacast_receiver.yaml"
)
add_dependencies(deltacast_receiver deltacast_receiver_yaml)
