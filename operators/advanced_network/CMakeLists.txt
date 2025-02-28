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

# Create targets for the holoscan-agnostic networking libraries.
# advanced_network_common will be generated and used by the operators below.
add_subdirectory(advanced_network)

# Find the holoscan SDK
find_package(holoscan 2.6 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# RX holoscan operator
add_library(advanced_network_rx SHARED adv_network_rx.cpp)
add_library(holoscan::ops::advanced_network_rx ALIAS advanced_network_rx)
target_link_libraries(advanced_network_rx PUBLIC advanced_network_common holoscan::core)
target_include_directories(advanced_network_rx
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)
set_target_properties(advanced_network_rx PROPERTIES
    OUTPUT_NAME "holoscan_op_advanced_network_rx"
    EXPORT_NAME ops::advanced_network_rx
)

# TX holoscan operator
add_library(advanced_network_tx SHARED adv_network_tx.cpp)
add_library(holoscan::ops::advanced_network_tx ALIAS advanced_network_tx)
target_link_libraries(advanced_network_tx PUBLIC advanced_network_common holoscan::core)
target_include_directories(advanced_network_tx
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)
set_target_properties(advanced_network_tx PROPERTIES
    OUTPUT_NAME "holoscan_op_advanced_network_tx"
    EXPORT_NAME ops::advanced_network_tx
)

# Installation
install(
    TARGETS
        advanced_network_rx
        advanced_network_tx
    EXPORT holoscan-networking-targets
    COMPONENT advanced_network-cpp
)
install(
    FILES
        adv_network_rx.h
        adv_network_tx.h
    DESTINATION include/holoscan/operators/advanced_network
    COMPONENT advanced_network-cpp
)

# Python bindings
if(HOLOHUB_BUILD_PYTHON)
    add_subdirectory(python)
endif()
