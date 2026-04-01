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

# Image processor GPU-resident operator
add_library(image_processor_gpu_resident SHARED
  image_processor_gpu_resident.cpp
)

add_library(holoscan::ops::image_processor_gpu_resident ALIAS image_processor_gpu_resident)

target_include_directories(image_processor_gpu_resident
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(image_processor_gpu_resident
  PUBLIC
    hololink::core
    hololink::common
    holoscan::core
    fmt::fmt
    CUDA::cuda_driver
)

target_compile_definitions(image_processor_gpu_resident
  PRIVATE HOLOSCAN_MAJOR_VERSION=${holoscan_VERSION_MAJOR}
  PRIVATE HOLOSCAN_MINOR_VERSION=${holoscan_VERSION_MINOR}
)

install(TARGETS image_processor_gpu_resident
  COMPONENT holoscan-ops
)
