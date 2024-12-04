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

message(STATUS "PROJECT_NAME: ${PROJECT_NAME}")

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_sources(${PROJECT_NAME} PRIVATE adv_network_dpdk_mgr.cpp)

pkg_check_modules(DPDK QUIET libdpdk)

if(NOT DPDK_FOUND)
  message(STATUS "Looking for DPDK in alternative directories")
  set(ENV{PKG_CONFIG_PATH} /opt/mellanox/dpdk/lib/x86_64-linux-gnu/pkgconfig/)
  pkg_check_modules(DPDK REQUIRED libdpdk)

  target_link_directories(${PROJECT_NAME} PUBLIC ${DPDK_LIBRARY_DIRS})
  target_include_directories(${PROJECT_NAME} PUBLIC ${DPDK_INCLUDE_DIRS})
  target_compile_options(${PROJECT_NAME} PUBLIC ${DPDK_CFLAGS_OTHER})
  target_link_libraries(${PROJECT_NAME} PUBLIC holoscan::core ${DPDK_LDFLAGS_OTHER} ${DPDK_LIBRARIES})
else() # Upstream DPDK
  set(DPDK_EXTRA_LIBS -Wl,--no-whole-archive -lmlx5 -libverbs -pthread -lnuma -ldl)
  target_link_libraries(${PROJECT_NAME} PUBLIC holoscan::core -L${DPDK_LIBRARY_DIRS} ${DPDK_EXTRA_LIBS} ${DPDK_LIBRARIES})
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "80;90")


target_compile_options(${PROJECT_NAME}  PUBLIC ${DPDK_CFLAGS})
