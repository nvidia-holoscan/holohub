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

# --- Dependencies ---

pkg_check_modules(DOCA REQUIRED doca-gpunetio doca-eth doca-flow)
pkg_check_modules(DPDK REQUIRED libdpdk)

# --- Sources ---

target_sources(${PROJECT_NAME}
  PRIVATE
    adv_network_doca_mgr.cpp
    adv_network_doca_mgr_obj.cpp
    adv_network_doca_kernels.cu
)

# --- Compile settings ---

target_include_directories(${PROJECT_NAME}
  PUBLIC ${DPDK_INCLUDE_DIRS}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(${PROJECT_NAME}
  PUBLIC
    ${DPDK_CFLAGS}
    ${DOCA_CFLAGS}
)

target_compile_definitions(${PROJECT_NAME}
  PUBLIC
    ALLOW_EXPERIMENTAL_API
    DOCA_ALLOW_EXPERIMENTAL_API
)

set_target_properties(${PROJECT_NAME}
  PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "80;90"
)

# --- Linking settings---

# To find the doca libraries below with just -l
target_link_directories(${PROJECT_NAME} PUBLIC ${DOCA_LIBRARY_DIRS})

target_link_libraries(${PROJECT_NAME}
  PRIVATE
    holoscan::core
    CUDA::cudart
    CUDA::cuda_driver
    -ldoca_gpunetio
    -ldoca_gpunetio_device # static
    -ldoca_eth
    -ldoca_flow
)
