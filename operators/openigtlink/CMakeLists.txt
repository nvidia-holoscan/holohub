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
cmake_minimum_required(VERSION 3.20)
project(openigtlink)

find_package(holoscan 1.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

find_package(OpenIGTLink REQUIRED PATHS "/workspace/OpenIGTLink-build")

add_library(openigtlink_rx SHARED
  openigtlink_rx.cpp
)
add_library(holoscan::ops::openigtlink_rx ALIAS openigtlink_rx)
target_link_libraries(openigtlink_rx
  PUBLIC
    holoscan::core
    holoscan::ops::holoviz
    OpenIGTLink
)
target_include_directories(openigtlink_rx INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_library(openigtlink_tx SHARED
  openigtlink_tx.cpp
)
add_library(holoscan::ops::openigtlink_tx ALIAS openigtlink_tx)
target_link_libraries(openigtlink_tx
  PUBLIC
    holoscan::core
    holoscan::ops::holoviz
    OpenIGTLink
)
target_include_directories(openigtlink_tx INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

if(HOLOHUB_BUILD_PYTHON)
    add_subdirectory(python)
endif()
