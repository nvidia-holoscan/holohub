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
target_sources(${PROJECT_NAME} PRIVATE
  adv_network_doca_mgr.cpp
  adv_network_doca_mgr_obj.cpp
  adv_network_doca_kernels.cu
)

pkg_check_modules(DOCA REQUIRED doca-gpunetio)
pkg_check_modules(DOCA REQUIRED doca-flow)
pkg_check_modules(DOCA REQUIRED doca-common)
pkg_check_modules(DOCA REQUIRED doca-argp)

pkg_check_modules(DPDK REQUIRED libdpdk)

target_include_directories(${PROJECT_NAME} PUBLIC ${DPDK_INCLUDE_DIRS})
target_compile_options(${PROJECT_NAME}  PUBLIC ${DPDK_CFLAGS})

target_compile_definitions(${PROJECT_NAME} PUBLIC ALLOW_EXPERIMENTAL_API)
target_compile_definitions(${PROJECT_NAME} PUBLIC DOCA_ALLOW_EXPERIMENTAL_API)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "80;90")

target_compile_options(${PROJECT_NAME} PUBLIC ${DOCA_CFLAGS})

target_link_libraries(${PROJECT_NAME}  PRIVATE holoscan::core)
target_link_libraries(${PROJECT_NAME} PRIVATE -L${DOCA_LIBRARY_DIRS} -ldoca_gpunetio libdoca_gpunetio_device.a -ldoca_common -ldoca_argp -ldoca_eth -ldoca_flow)
