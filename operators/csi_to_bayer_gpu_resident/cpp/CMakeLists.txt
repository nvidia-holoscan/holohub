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

find_package(holoscan 4.0.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")
find_package(Hololink 2.5 REQUIRED CONFIG
             PATHS "/opt/nvidia/hololink")

# CSI-to-Bayer converter base library
add_library(csi_to_bayer_converter_base STATIC
  csi_to_bayer_converter_base.cpp
)
set_property(TARGET csi_to_bayer_converter_base PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(csi_to_bayer_converter_base
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(csi_to_bayer_converter_base
  PUBLIC
    hololink::core
    holoscan::core
    fmt::fmt
)

# CSI-to-Bayer GPU-resident operator
add_library(csi_to_bayer_gpu_resident SHARED
  csi_to_bayer_gpu_resident.cpp
)

add_library(holoscan::ops::csi_to_bayer_gpu_resident ALIAS csi_to_bayer_gpu_resident)

target_include_directories(csi_to_bayer_gpu_resident
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(csi_to_bayer_gpu_resident
  PUBLIC
    csi_to_bayer_converter_base
    hololink::core
    hololink::common
    holoscan::core
    fmt::fmt
    CUDA::cuda_driver
)

target_compile_definitions(csi_to_bayer_gpu_resident
  PRIVATE HOLOSCAN_MAJOR_VERSION=${holoscan_VERSION_MAJOR}
  PRIVATE HOLOSCAN_MINOR_VERSION=${holoscan_VERSION_MINOR}
)

install(TARGETS csi_to_bayer_gpu_resident csi_to_bayer_converter_base
  COMPONENT holoscan-ops
)
