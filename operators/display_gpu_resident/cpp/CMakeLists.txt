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

enable_language(CUDA)

add_library(display_gpu_resident SHARED
  display_gpu_resident.cpp
  display_kernels.cu
)

add_library(holoscan::ops::display_gpu_resident ALIAS display_gpu_resident)

set_target_properties(display_gpu_resident PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_include_directories(display_gpu_resident
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../cudisp/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(display_gpu_resident
  PUBLIC
    holoscan::core
    cuDisp
    CUDA::cuda_driver
)

target_compile_definitions(display_gpu_resident
  PRIVATE HOLOSCAN_MAJOR_VERSION=${holoscan_VERSION_MAJOR}
  PRIVATE HOLOSCAN_MINOR_VERSION=${holoscan_VERSION_MINOR}
)

install(TARGETS display_gpu_resident
  COMPONENT holoscan-ops
)
