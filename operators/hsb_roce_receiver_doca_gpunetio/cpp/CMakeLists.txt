# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

find_package(holoscan 4.0.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

find_package(Hololink 2.5 REQUIRED CONFIG
    PATHS "/opt/nvidia/hololink")

# libibverbs
find_package(PkgConfig REQUIRED)
pkg_check_modules(IBVERBS REQUIRED libibverbs)

# DOCA SDK (aarch64-only)
set(DOCA_PATH "/opt/mellanox/doca")
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(DOCA_PKGCONFIG_DIR "${DOCA_PATH}/lib/aarch64-linux-gnu/pkgconfig")
else()
    message(WARNING "hsb_roce_receiver_doca_gpunetio is only supported on aarch64 "
                    "(detected ${CMAKE_SYSTEM_PROCESSOR}). Skipping.")
    return()
endif()
set(ENV{PKG_CONFIG_PATH} "${DOCA_PKGCONFIG_DIR}:$ENV{PKG_CONFIG_PATH}")

pkg_check_modules(DOCA doca-verbs doca-gpunetio doca-common)
if(NOT DOCA_FOUND)
  message(WARNING "DOCA SDK not found (checked ${DOCA_PKGCONFIG_DIR}). "
                  "hsb_roce_receiver_doca_gpunetio will not be built.")
  return()
endif()

message(STATUS "DOCA found")
message(STATUS "  include:   ${DOCA_INCLUDE_DIRS}")
message(STATUS "  libraries: ${DOCA_LINK_LIBRARIES}")

add_library(hsb_roce_receiver_doca_gpunetio SHARED
  doca_roce_receiver_op.cpp
  doca_helpers.cpp
  doca_roce_receiver_kernel.cu
)

add_library(holoscan::ops::hsb_roce_receiver_doca_gpunetio
            ALIAS hsb_roce_receiver_doca_gpunetio)

set_property(TARGET hsb_roce_receiver_doca_gpunetio PROPERTY CUDA_ARCHITECTURES native)

target_include_directories(hsb_roce_receiver_doca_gpunetio
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    ${IBVERBS_INCLUDE_DIRS}
    ${DOCA_INCLUDE_DIRS}
)

target_link_libraries(hsb_roce_receiver_doca_gpunetio
  PUBLIC
    hololink::core
    hololink::common
    holoscan::core
    fmt::fmt
    CUDA::cuda_driver
    ${IBVERBS_LIBRARIES}
  PRIVATE
    ${DOCA_LINK_LIBRARIES}
)

target_compile_options(hsb_roce_receiver_doca_gpunetio PRIVATE -Wno-deprecated-declarations)

set_target_properties(hsb_roce_receiver_doca_gpunetio PROPERTIES
  INSTALL_RPATH "$ORIGIN;${DOCA_LIBRARY_DIRS}"
  BUILD_RPATH "${DOCA_LIBRARY_DIRS}"
)

target_compile_definitions(hsb_roce_receiver_doca_gpunetio
  PRIVATE HOLOSCAN_MAJOR_VERSION=${holoscan_VERSION_MAJOR}
  PRIVATE HOLOSCAN_MINOR_VERSION=${holoscan_VERSION_MINOR}
)

install(TARGETS hsb_roce_receiver_doca_gpunetio
  COMPONENT holoscan-ops
)
