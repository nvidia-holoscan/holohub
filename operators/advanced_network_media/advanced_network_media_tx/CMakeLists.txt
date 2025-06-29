# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(advanced_network_media_tx)

find_package(holoscan 2.6 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_library(${PROJECT_NAME} SHARED
    adv_network_media_tx.cpp
)

add_library(holoscan::ops::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../advanced_network
    ${CMAKE_CURRENT_SOURCE_DIR}/../../advanced_network/advanced_network
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
    holoscan::core
    GXF::multimedia
    rivermax-dev-kit
    advanced_network_common
    advanced_network_media_common
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    OUTPUT_NAME "holoscan_op_advanced_network_media_tx"
    EXPORT_NAME ops::advanced_network_media_tx
)

# Installation
install(
    TARGETS
    ${PROJECT_NAME}
    EXPORT holoscan-networking-targets
    COMPONENT advanced_network-cpp
)

if(HOLOHUB_BUILD_PYTHON)
    add_subdirectory(python)
endif()
