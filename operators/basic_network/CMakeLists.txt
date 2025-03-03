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

# Create the aliases with the desired naming scheme
add_library(holoscan::ops::basic_network ALIAS basic_network)
target_link_libraries(basic_network holoscan::core)
target_include_directories(basic_network
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
        $<INSTALL_INTERFACE:include/holoscan/operators/basic_network>
        # We'd prefer to avoid this last line, so that developers include the operator headers with
        # #include "holoscan/operators/basic_network/basic_network_*x.h". However, this is needed
        # for basic_networking_ping to find the operator in the install tree without modifying the
        # #include of basic_network_*x.h in the application source code in holohub.
)
set_target_properties(basic_network PROPERTIES
    OUTPUT_NAME "holoscan_op_basic_network"
    EXPORT_NAME ops::basic_network
)

# Installation
install(
    TARGETS basic_network
    EXPORT holoscan-networking-targets
    COMPONENT basic_network-cpp
)

install(
    FILES
      basic_network_operator_tx.h
      basic_network_operator_rx.h
      basic_network_operator_common.h
    DESTINATION include/holoscan/operators/basic_network
    COMPONENT basic_network-cpp
)

# Python equivalent
if(HOLOHUB_BUILD_PYTHON)
  add_subdirectory(python)
endif()
