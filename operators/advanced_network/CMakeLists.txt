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

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --diag-suppress 1217")
set(CMAKE_CUDA_ARCHITECTURES "80;90")

#set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -ggdb")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -ggdb")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")


# Option to enforce the shared CUDA runtime library usage
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --cudart=shared -Xcompiler -fvisibility=hidden")
#set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

add_compile_definitions(ALLOW_EXPERIMENTAL_API)
add_compile_definitions(DOCA_ALLOW_EXPERIMENTAL_API)

execute_process(COMMAND arch OUTPUT_VARIABLE ARCH)
string(STRIP ${ARCH} ARCH)
set(DPDK_PATH /opt/mellanox/dpdk)
set(DOCA_PATH /opt/mellanox/doca)
set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:${DPDK_PATH}/lib/${ARCH}-linux-gnu/pkgconfig:${DOCA_PATH}/lib/${ARCH}-linux-gnu/pkgconfig:/opt/mellanox/flexio/lib/pkgconfig:/opt/mellanox/collectx/lib/${ARCH}-linux-gnu/pkgconfig")

# Initialize ANO_MGR with a default if not provided
if(NOT DEFINED ANO_MGR)
    set(ANO_MGR "dpdk doca" CACHE STRING "Manager type(s) list")
endif()

separate_arguments(ANO_MGR_LIST UNIX_COMMAND ${ANO_MGR})

add_subdirectory(managers)

add_library(advanced_network_common SHARED
  adv_network_common.cpp
  adv_network_kernels.cu
  managers/adv_network_mgr.cpp
)

# Define targets for each manager type
foreach(MGR IN LISTS ANO_MGR_LIST)
    string(TOUPPER ${MGR} MGR_UPPER)
    string(TOLOWER ${MGR} MGR_LOWER)
    message(STATUS "Adding manager ${MGR_UPPER} to the configuration")
    target_compile_definitions(advanced_network_common PUBLIC "ANO_MGR_${MGR_UPPER}=1")
    target_link_libraries(advanced_network_common PRIVATE ano_manager_${MGR_LOWER})
endforeach()

target_include_directories(advanced_network_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/managers)

add_library(advanced_network_rx SHARED adv_network_rx.cpp)
add_library(advanced_network_tx SHARED adv_network_tx.cpp)

add_library(holoscan::advanced_network_rx ALIAS advanced_network_rx)
add_library(holoscan::advanced_network_tx ALIAS advanced_network_tx)


set_target_properties(advanced_network_common PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(advanced_network_common PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(advanced_network_common PROPERTIES CUDA_ARCHITECTURES "80;90")
set_target_properties(advanced_network_rx PROPERTIES CUDA_ARCHITECTURES "80;90")
set_target_properties(advanced_network_tx PROPERTIES CUDA_ARCHITECTURES "80;90")


target_link_libraries(advanced_network_common PUBLIC holoscan::core)
target_link_libraries(advanced_network_rx PUBLIC advanced_network_common)
target_link_libraries(advanced_network_tx PUBLIC advanced_network_common)

if(HOLOHUB_BUILD_PYTHON)
    add_subdirectory(python)
endif()

# Installation
install(TARGETS advanced_network_common
                advanced_network_rx
                advanced_network_tx)
