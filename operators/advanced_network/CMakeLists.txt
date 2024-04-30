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
project(advanced_network)

find_package(holoscan 0.6 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")
find_package(PkgConfig)

enable_language(CUDA)

add_library(advanced_network_common SHARED
  adv_network_common.cpp
  adv_network_kernels.cu
  managers/adv_network_mgr.cpp
  managers/dpdk/adv_network_dpdk_mgr.cpp
)

target_include_directories(advanced_network_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/managers ${CMAKE_CURRENT_SOURCE_DIR}/managers/dpdk ${CMAKE_CURRENT_SOURCE_DIR})

add_library(advanced_network_rx SHARED adv_network_rx.cpp)
add_library(advanced_network_tx SHARED adv_network_tx.cpp)

add_library(holoscan::advanced_network_rx ALIAS advanced_network_rx)
add_library(holoscan::advanced_network_tx ALIAS advanced_network_tx)

pkg_check_modules(DPDK QUIET libdpdk)

if(NOT DPDK_FOUND)
  message(STATUS "Looking for DPDK in alternative directories")
  set(ENV{PKG_CONFIG_PATH} /opt/mellanox/dpdk/lib/x86_64-linux-gnu/pkgconfig/)
  pkg_check_modules(DPDK REQUIRED libdpdk)

  target_link_directories(advanced_network_common PUBLIC ${DPDK_LIBRARY_DIRS})
  target_include_directories(advanced_network_common PUBLIC ${DPDK_INCLUDE_DIRS})
  target_compile_options(advanced_network_common PUBLIC ${DPDK_CFLAGS_OTHER})
  target_link_libraries(advanced_network_common PUBLIC ${DPDK_LDFLAGS_OTHER} ${DPDK_LIBRARIES})
else() # Upstream DPDK
  set(DPDK_EXTRA_LIBS -Wl,--no-whole-archive -lmlx5 -libverbs -pthread -lnuma -ldl)
  target_link_libraries(advanced_network_common PUBLIC ${DPDK_EXTRA_LIBS} ${DPDK_LIBRARIES})
endif()

target_compile_options(advanced_network_common PUBLIC ${DPDK_CFLAGS})
target_compile_options(advanced_network_rx PUBLIC ${DPDK_CFLAGS})
target_compile_options(advanced_network_tx PUBLIC ${DPDK_CFLAGS})

target_link_libraries(advanced_network_common PUBLIC holoscan::core)
target_link_libraries(advanced_network_rx PUBLIC advanced_network_common)
target_link_libraries(advanced_network_tx PUBLIC advanced_network_common)

if(HOLOHUB_BUILD_PYTHON)
    add_subdirectory(python)
endif()
