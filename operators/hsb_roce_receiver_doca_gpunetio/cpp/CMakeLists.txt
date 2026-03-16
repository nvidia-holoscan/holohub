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

# DOCA SDK
set(TARGET_ARCH "arm64")
if(DEFINED ENV{TARGET_ARCH})
  set(TARGET_ARCH "$ENV{TARGET_ARCH}")
endif()

set(DOCA_PATH "/opt/mellanox/doca")

if(TARGET_ARCH STREQUAL "amd64" OR TARGET_ARCH STREQUAL "x86_64")
    set(DOCA_LIB_DIR "${DOCA_PATH}/lib/x86_64-linux-gnu")
elseif(TARGET_ARCH STREQUAL "arm64" OR TARGET_ARCH STREQUAL "aarch64")
    set(DOCA_LIB_DIR "${DOCA_PATH}/lib/aarch64-linux-gnu")
else()
    set(DOCA_LIB_DIR "${DOCA_PATH}/lib")
endif()

find_path(DOCA_INCLUDE_DIR doca_verbs.h
  PATHS ${DOCA_PATH}/include
  NO_DEFAULT_PATH
)
find_path(DOCA_DEV_INCLUDE_DIR doca_dev.h
  PATHS ${DOCA_PATH}/include
  NO_DEFAULT_PATH
)
find_path(DOCA_GPUNETIO_INCLUDE_DIR doca_gpunetio.h
  PATHS ${DOCA_PATH}/include
  NO_DEFAULT_PATH
)

find_library(DOCA_VERBS_LIB doca_verbs
  PATHS ${DOCA_LIB_DIR}
  NO_DEFAULT_PATH
)

find_library(DOCA_GPUNETIO_LIB doca_gpunetio
  PATHS ${DOCA_LIB_DIR}
  NO_DEFAULT_PATH
)

find_library(DOCA_COMMON_LIB doca_common
  PATHS ${DOCA_LIB_DIR}
  NO_DEFAULT_PATH
)

if(NOT (DOCA_INCLUDE_DIR AND DOCA_DEV_INCLUDE_DIR AND DOCA_GPUNETIO_INCLUDE_DIR
        AND DOCA_VERBS_LIB AND DOCA_COMMON_LIB AND DOCA_GPUNETIO_LIB))
  message(WARNING "DOCA SDK not found at ${DOCA_PATH}. "
                  "hsb_roce_receiver_doca_gpunetio will not be built.")
  return()
endif()

message(STATUS "DOCA found at ${DOCA_PATH}")
message(STATUS "  include: ${DOCA_INCLUDE_DIR}")
message(STATUS "  verbs:   ${DOCA_VERBS_LIB}")
message(STATUS "  gpunetio: ${DOCA_GPUNETIO_LIB}")
message(STATUS "  common:  ${DOCA_COMMON_LIB}")

add_library(hsb_roce_receiver_doca_gpunetio SHARED
  doca_roce_receiver_op.cpp
  doca_helpers.cpp
  doca_roce_receiver_kernel.cu
)

add_library(holoscan::ops::hsb_roce_receiver_doca_gpunetio
            ALIAS hsb_roce_receiver_doca_gpunetio)

# Build DOCA receiver kernels for Ampere RTX A6000 (sm_86) and newer GPUs.
set_property(TARGET hsb_roce_receiver_doca_gpunetio PROPERTY CUDA_ARCHITECTURES 86 87 90)

target_include_directories(hsb_roce_receiver_doca_gpunetio
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${IBVERBS_INCLUDE_DIRS}
    ${DOCA_INCLUDE_DIR}
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
    ${DOCA_VERBS_LIB}
    ${DOCA_GPUNETIO_LIB}
    ${DOCA_COMMON_LIB}
)

target_compile_options(hsb_roce_receiver_doca_gpunetio PRIVATE -Wno-deprecated-declarations)

set_target_properties(hsb_roce_receiver_doca_gpunetio PROPERTIES
  INSTALL_RPATH "$ORIGIN:${DOCA_LIB_DIR}"
  BUILD_RPATH "${DOCA_LIB_DIR}"
)

target_compile_definitions(hsb_roce_receiver_doca_gpunetio
  PRIVATE HOLOSCAN_MAJOR_VERSION=${holoscan_VERSION_MAJOR}
  PRIVATE HOLOSCAN_MINOR_VERSION=${holoscan_VERSION_MINOR}
)

install(TARGETS hsb_roce_receiver_doca_gpunetio
  COMPONENT holoscan-ops
)
