# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(basic_networking_ping CXX)

find_package(holoscan 0.5 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_executable(basic_networking_ping
  main.cpp
)

target_link_libraries(basic_networking_ping
  PRIVATE
  holoscan::core
  basic_network
)

# Copy config file
add_custom_target(basic_networking_ping_rx_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/../basic_networking_ping_rx.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../basic_networking_ping_rx.yaml"
)
add_custom_target(basic_networking_ping_tx_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/../basic_networking_ping_tx.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../basic_networking_ping_tx.yaml"
)

add_dependencies(basic_networking_ping basic_networking_ping_rx_yaml)
add_dependencies(basic_networking_ping basic_networking_ping_tx_yaml)

install(
  TARGETS basic_networking_ping
  DESTINATION examples/basic_networking_ping
  COMPONENT basic_networking_ping-cpp
)

install(
  FILES
    ../basic_networking_ping_rx.yaml
    ../basic_networking_ping_tx.yaml
  DESTINATION examples/basic_networking_ping
  COMPONENT basic_networking_ping-configs
  PERMISSIONS OWNER_READ OWNER_WRITE
              GROUP_READ GROUP_WRITE
              WORLD_READ WORLD_WRITE
)

install(
  FILES CMakeLists.txt.install
  RENAME CMakeLists.txt
  DESTINATION examples/basic_networking_ping
  COMPONENT basic_networking_ping-cppsrc
  PERMISSIONS OWNER_READ OWNER_WRITE
              GROUP_READ GROUP_WRITE
              WORLD_READ WORLD_WRITE
)

install(
  FILES
    main.cpp
  DESTINATION examples/basic_networking_ping
  COMPONENT basic_networking_ping-cppsrc
  PERMISSIONS OWNER_READ OWNER_WRITE
              GROUP_READ GROUP_WRITE
              WORLD_READ WORLD_WRITE
)
