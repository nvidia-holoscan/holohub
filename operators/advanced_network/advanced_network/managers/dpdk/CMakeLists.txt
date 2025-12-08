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

message(STATUS "PROJECT_NAME: ${PROJECT_NAME}")

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_sources(${PROJECT_NAME} PRIVATE adv_network_dpdk_mgr.cpp adv_network_dpdk_stats.cpp)

# Try to find DPDK via pkg-config
# This will find upstream DPDK (installed to /usr/local) or mlnx-dpdk
pkg_check_modules(DPDK QUIET libdpdk)

if(NOT DPDK_FOUND)
  # Fallback: try to find mlnx-dpdk in standard Mellanox installation paths
  message(STATUS "Looking for DPDK in Mellanox installation directories")
  # Try to find DPDK via pkg-config
  # Note: adding the following so it can find both upstream DPDK (installed to /usr/local) but also mlnx-dpdk (in `/opt/mellanox/dpdk`)
  list(APPEND CMAKE_PREFIX_PATH "/opt/mellanox/dpdk/lib/${CMAKE_HOST_SYSTEM_PROCESSOR}-linux-gnu/")
  pkg_check_modules(DPDK REQUIRED libdpdk)
  message(STATUS "Found DPDK: ${DPDK_VERSION} in ${DPDK_LIBRARY_DIRS}")
endif()

# Common DPDK configuration
target_link_directories(${PROJECT_NAME} PUBLIC ${DPDK_LIBRARY_DIRS})
target_include_directories(${PROJECT_NAME} PUBLIC ${DPDK_INCLUDE_DIRS})
target_compile_options(${PROJECT_NAME} PUBLIC ${DPDK_CFLAGS_OTHER})

# Link libraries - add extra libs for Mellanox NIC support
set(DPDK_EXTRA_LIBS -Wl,--no-whole-archive -lmlx5 -libverbs -pthread -lnuma -ldl)
target_link_libraries(${PROJECT_NAME}
  PUBLIC
    ${DPDK_LDFLAGS_OTHER}
    ${DPDK_LIBRARIES}
    ${DPDK_EXTRA_LIBS}
  PRIVATE
    holoscan::core
    CUDA::cudart
    CUDA::cuda_driver
)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "80;90")


target_compile_options(${PROJECT_NAME}  PUBLIC ${DPDK_CFLAGS})
