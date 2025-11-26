# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(advanced_network)


find_package(holoscan 2.6 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install") # Remove if holoscan logger dependency removed

find_package(PkgConfig)
find_package(CUDAToolkit REQUIRED)

enable_language(CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --diag-suppress 1217")
set(CMAKE_CUDA_ARCHITECTURES "80;90")

add_compile_definitions(ALLOW_EXPERIMENTAL_API)
add_compile_definitions(DOCA_ALLOW_EXPERIMENTAL_API)

execute_process(COMMAND arch OUTPUT_VARIABLE ARCH)
string(STRIP ${ARCH} ARCH)

# Path configurations for DPDK and DOCA
# Upstream DPDK (installed via meson) uses /usr/local by default
# DOCA/mlnx-dpdk uses /opt/mellanox paths
set(DPDK_PATH_UPSTREAM /usr/local)
set(DPDK_PATH_MLNX /opt/mellanox/dpdk)
set(DOCA_PATH /opt/mellanox/doca)

# Set PKG_CONFIG_PATH to find DPDK (upstream first, then mlnx) and DOCA
set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:${DPDK_PATH_UPSTREAM}/lib/${ARCH}-linux-gnu/pkgconfig:${DPDK_PATH_UPSTREAM}/lib/pkgconfig:${DPDK_PATH_MLNX}/lib/${ARCH}-linux-gnu/pkgconfig:${DOCA_PATH}/lib/${ARCH}-linux-gnu/pkgconfig:/opt/mellanox/flexio/lib/pkgconfig:/opt/mellanox/collectx/lib/${ARCH}-linux-gnu/pkgconfig")

# Common library
add_library(advanced_network_common SHARED
  common.cpp
  kernels.cu
  manager.cpp
)
target_include_directories(advanced_network_common
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..> # so we can find it under #include "advanced_network/..."
        $<INSTALL_INTERFACE:include>
)
add_library(holoscan::advanced_network ALIAS advanced_network_common)
target_link_libraries(advanced_network_common PUBLIC holoscan::core) # Remove if holoscan logger dependency removed
set_target_properties(advanced_network_common PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    OUTPUT_NAME "holoscan_advanced_network"
    EXPORT_NAME "advanced_network"
)
install(
    TARGETS advanced_network_common
    EXPORT holoscan-networking-targets
    COMPONENT advanced_network-cpp
)
install(
    FILES
        common.h
        kernels.h
        types.h
    DESTINATION include/advanced_network
    COMPONENT advanced_network-cpp
)

# Backend libraries
if(NOT DEFINED ANO_MGR)
    # Initialize ANO_MGR with a default if not provided
    # Default: dpdk only (no DOCA dependency)
    # Add "gpunetio" to enable GPU-accelerated networking (requires DOCA)
    # Note: rivermax not added to the default list since it requires licensing
    set(ANO_MGR "dpdk" CACHE STRING "Manager type(s) list")
endif()
separate_arguments(ANO_MGR_LIST UNIX_COMMAND ${ANO_MGR})

# Define targets for each manager type
foreach(MGR IN LISTS ANO_MGR_LIST)
    string(TOUPPER ${MGR} MGR_UPPER)
    string(TOLOWER ${MGR} MGR_LOWER)

    message(STATUS "Adding manager ${MGR_UPPER} to the configuration")
    project(advanced_network_${MGR_LOWER} LANGUAGES CXX CUDA)
    add_library(${PROJECT_NAME} SHARED)
    add_subdirectory(managers/${MGR_LOWER})
    target_compile_definitions(${PROJECT_NAME} PUBLIC "ANO_MGR_${MGR_UPPER}=1")
    target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        OUTPUT_NAME "holoscan_advanced_network_${MGR_LOWER}"
        EXPORT_NAME "advanced_network::${MGR_LOWER}"
    )
    install(TARGETS ${PROJECT_NAME}
        EXPORT holoscan-networking-targets
        COMPONENT advanced_network-cpp
    )

    target_compile_definitions(advanced_network_common PUBLIC "ANO_MGR_${MGR_UPPER}=1")
    target_link_libraries(advanced_network_common PRIVATE advanced_network_${MGR_LOWER})
endforeach()
