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
project(volume_renderer)

find_package(holoscan 0.6 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# we don't need the gRPC interface, OpenH264 and the examples
set(CLARA_VIZ_WITH_GRPC OFF CACHE INTERNAL "")
set(CLARA_VIZ_WITH_OPENH264 OFF CACHE INTERNAL "")
set(CLARA_VIZ_WITH_EXAMPLES OFF CACHE INTERNAL "")

# we are downloading large content, it's better to see the progress
set(FETCHCONTENT_QUIET OFF)

# Set CMP0135 policy to NEW to use time of extraction for files extracted by
# FetchContent/ExternalProject_Add.
# This ensures that anything that depends on the extracted contents will be rebuilt whenever
# the URL changes.
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
  set(CMAKE_POLICY_DEFAULT_CMP0135 NEW)
endif()

# set CMP0148 policy to OLD to allow FindCUDA to be used by ClaraViz
if(POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD)
  set(CMAKE_POLICY_DEFAULT_CMP0146 OLD)
endif()

set(_clara_viz_version "0.3.1")
include(FetchContent)
FetchContent_Declare(
    ClaraViz
    URL https://github.com/NVIDIA/clara-viz/archive/refs/tags/v${_clara_viz_version}.tar.gz
    URL_MD5 8cc1a29481df378a9413bfe1c3083de4
    )

# enable CMP0077 to allow overwriting option() statements in FetchContent sub-projects
cmake_policy(SET CMP0077 NEW)

FetchContent_GetProperties(ClaraViz)
if(NOT claraviz_POPULATED)
    FetchContent_Populate(ClaraViz)
    set(CLARA_VIZ_PUBLIC_CMAKE_TOOLS_DIR "${claraviz_SOURCE_DIR}/cmake")
    add_subdirectory(${claraviz_SOURCE_DIR}/thirdparty ${claraviz_BINARY_DIR}/thirdpary)
    add_subdirectory(${claraviz_SOURCE_DIR}/src ${claraviz_BINARY_DIR}/src)
endif()
set(FETCHCONTENT_QUIET ON)

set(CLARA_VIZ_PUBLIC_CMAKE_TOOLS_DIR "${claraviz_SOURCE_DIR}/cmake")

find_package(clara_viz_renderer REQUIRED HINTS ${claraviz_SOURCE_DIR}/cmake)
install(IMPORTED_RUNTIME_ARTIFACTS clara::viz::renderer
  COMPONENT "volume_renderer"
)

add_library(volume_renderer SHARED
  dataset.cpp
  dataset.hpp
  video_buffer_blob.hpp
  volume_renderer.cpp
  volume_renderer.hpp
  )

add_library(holoscan::ops::volume_renderer ALIAS volume_renderer)

target_link_libraries(volume_renderer
  PRIVATE
    holoscan::core
    clara::viz::renderer
    clara::viz::core
    clara::viz::core::hw
)

target_include_directories(volume_renderer INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(volume_renderer PRIVATE HOLOSCAN_MAJOR_VERSION=${holoscan_VERSION_MAJOR})
target_compile_definitions(volume_renderer PRIVATE HOLOSCAN_MINOR_VERSION=${holoscan_VERSION_MINOR})

install(
  TARGETS volume_renderer
  EXPORT holoscan-ops
)
