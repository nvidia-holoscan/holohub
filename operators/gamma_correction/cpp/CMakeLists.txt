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

# Add gamma_correction operator
add_library(gamma_correction SHARED
  gamma_correction.cpp
)

add_library(holoscan::ops::gamma_correction ALIAS gamma_correction)

# generate header from Holoscan slang shader
include(GenHeaderFromFile)
gen_header_from_file(TARGET gamma_correction FILE_PATH "gamma_correction.slang" TYPE text)

target_include_directories(gamma_correction
  PRIVATE
    ${slang_SOURCE_DIR}/include
    # for the generated header
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

target_link_libraries(gamma_correction
  PUBLIC
    holoscan::ops::slang_shader
)

target_include_directories(gamma_correction PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(gamma_correction PRIVATE HOLOSCAN_MAJOR_VERSION=${holoscan_VERSION_MAJOR})
target_compile_definitions(gamma_correction PRIVATE HOLOSCAN_MINOR_VERSION=${holoscan_VERSION_MINOR})

install(TARGETS gamma_correction
  COMPONENT holoscan-ops
)
