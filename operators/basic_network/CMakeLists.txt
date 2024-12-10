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
project(basic_network)

find_package(holoscan 0.5 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_library(basic_network SHARED
  basic_network_operator_tx.cpp
  basic_network_operator_rx.cpp
)

add_library(holoscan::basic_network ALIAS basic_network)
target_include_directories(basic_network PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(basic_network holoscan::core)

# Installation
install(
    TARGETS basic_network
    COMPONENT holoscan-networking
)

# Python equivalent
if(HOLOHUB_BUILD_PYTHON)
  add_subdirectory(python)
endif()
