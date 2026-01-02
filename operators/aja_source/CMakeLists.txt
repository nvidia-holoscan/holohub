# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(aja_source)

find_package(holoscan 1.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Fetch AJA nvt2 repository
include(FetchContent)
FetchContent_Declare(
    ajantv2
    GIT_REPOSITORY https://github.com/nvidia-holoscan/libajantv2.git
    GIT_TAG f74a9cab77a3bdb9f448b3eacc493fe63d363e9f
)

set(ENV{NTV2_VERSION_BUILD} 1)
set(AJANTV2_DISABLE_DEMOS ON)
set(AJANTV2_DISABLE_DRIVER ON)
set(AJANTV2_DISABLE_PLUGINS ON)
set(AJANTV2_DISABLE_TESTS ON)
set(AJANTV2_DISABLE_TOOLS ON)
set(AJA_INSTALL_HEADERS OFF)
set(AJA_INSTALL_SOURCES OFF)
message(STATUS "Fetching AJA sources")
FetchContent_MakeAvailable(ajantv2)
# Disable all the warnings for AJA
target_compile_options(ajantv2 PRIVATE -w)

add_library(AJA::ajantv2 ALIAS ajantv2)

add_library(aja_source SHARED
  aja_source.hpp
  ntv2channel.hpp
  aja_source.cpp
  )

add_library(holoscan::aja ALIAS aja_source)
add_library(holoscan::ops::aja ALIAS aja_source)

target_include_directories(aja_source INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(aja_source INTERFACE AJA_SOURCE)

target_link_libraries(aja_source
    PUBLIC
        holoscan::core
        AJA::ajantv2
        CUDA::cuda_driver
    PRIVATE
        CUDA::cudart
        GXF::multimedia
)

if(HOLOHUB_BUILD_PYTHON)
    add_subdirectory(python)
endif()

install(TARGETS aja_source)
