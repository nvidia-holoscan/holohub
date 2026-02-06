# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

target_sources(${PROJECT_NAME} PRIVATE adv_network_rdma_mgr.cpp)

pkg_check_modules(IBVERBS libibverbs)
pkg_check_modules(RDMACM librdmacm)

if(NOT IBVERBS_FOUND)
    find_library(IBVERBS_LIBRARIES
        NAMES ibverbs
        PATHS /usr/lib /usr/lib64
    )

    find_path(IBVERBS_INCLUDE_DIRS
        NAMES infiniband/verbs.h
        PATHS /usr/include
    )
endif()

if(NOT RDMACM_FOUND)
    find_library(RDMACM_LIBRARIES
        NAMES rdmacm
        PATHS /usr/lib /usr/lib64
    )
endif()

if(NOT IBVERBS_LIBRARIES OR NOT RDMACM_LIBRARIES)
    message(FATAL_ERROR "Required RDMA libraries not found")
endif()


# DPDK is needed for rings (only used in .cpp, not exposed in public headers)
pkg_check_modules(DPDK QUIET libdpdk)

if(NOT DPDK_FOUND)
  message(STATUS "Looking for DPDK in alternative directories")
  set(ENV{PKG_CONFIG_PATH} /opt/mellanox/dpdk/lib/x86_64-linux-gnu/pkgconfig/)
  pkg_check_modules(DPDK REQUIRED libdpdk)
endif()

target_link_directories(${PROJECT_NAME} PRIVATE ${DPDK_LIBRARY_DIRS})
target_include_directories(${PROJECT_NAME} PRIVATE ${DPDK_INCLUDE_DIRS})
target_compile_options(${PROJECT_NAME} PRIVATE ${DPDK_CFLAGS_OTHER})

# ibverbs/rdmacm: PUBLIC (adv_network_rdma_mgr.h includes <infiniband/verbs.h> and <rdma/rdma_cma.h>)
target_include_directories(${PROJECT_NAME} PUBLIC ${IBVERBS_INCLUDE_DIRS})
target_link_directories(${PROJECT_NAME} PUBLIC ${IBVERBS_LIBRARY_DIRS})

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        ${IBVERBS_LIBRARIES}
        ${RDMACM_LIBRARIES}
    PRIVATE
        holoscan::core
        ${DPDK_LDFLAGS_OTHER}
        ${DPDK_LIBRARIES}
)