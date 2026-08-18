# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

find_package(ZLIB REQUIRED)

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

# ClaraViz 0.4.2 requires deprecated FindCUDA support. Suppress the associated
# CMP0146 OLD warnings only while configuring ClaraViz.
set(_clara_viz_version "v0.4.2")
include(FetchContent)
FetchContent_Declare(
    ClaraViz
    URL https://github.com/NVIDIA/clara-viz/archive/refs/tags/${_clara_viz_version}.tar.gz
    URL_MD5 79cff506c896eb9d1a0d5b742782e06b
    )

# enable CMP0077 to allow overwriting option() statements in FetchContent sub-projects
cmake_policy(SET CMP0077 NEW)

# Allowing FetchContent_Populate() for now
# This should be updated when ClaraViz supports new version of CMake
if(POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)
endif()

function(_holohub_configure_claraviz)
  cmake_policy(PUSH)
  set(_suppress_dependency_warnings OFF)
  if(NOT DEFINED HOLOHUB_SUPPRESS_DEPENDENCY_DEPRECATION_WARNINGS OR
     HOLOHUB_SUPPRESS_DEPENDENCY_DEPRECATION_WARNINGS)
    set(_suppress_dependency_warnings ON)
  endif()
  if(_suppress_dependency_warnings AND CMAKE_VERSION VERSION_GREATER_EQUAL 4.4)
    cmake_policy(SET CMP0218 NEW)
    cmake_diagnostic(PUSH)
    cmake_diagnostic(SET CMD_DEPRECATED IGNORE)
    set(CMAKE_POLICY_DEFAULT_CMP0218 NEW)
  elseif(_suppress_dependency_warnings)
    set(CMAKE_WARN_DEPRECATED OFF)
  endif()
  if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
    set(CMAKE_POLICY_DEFAULT_CMP0146 OLD)
  endif()

  FetchContent_GetProperties(ClaraViz)
  if(NOT claraviz_POPULATED)
    FetchContent_Populate(ClaraViz)
    set(CLARA_VIZ_PUBLIC_CMAKE_TOOLS_DIR "${claraviz_SOURCE_DIR}/cmake")
    add_subdirectory(${claraviz_SOURCE_DIR}/thirdparty ${claraviz_BINARY_DIR}/thirdparty)
    add_subdirectory(${claraviz_SOURCE_DIR}/src ${claraviz_BINARY_DIR}/src)
  endif()

  set(claraviz_SOURCE_DIR "${claraviz_SOURCE_DIR}" PARENT_SCOPE)
  if(_suppress_dependency_warnings AND CMAKE_VERSION VERSION_GREATER_EQUAL 4.4)
    cmake_diagnostic(POP)
  endif()
  cmake_policy(POP)
endfunction()
_holohub_configure_claraviz()

set(CLARA_VIZ_PUBLIC_CMAKE_TOOLS_DIR "${claraviz_SOURCE_DIR}/cmake")
set(FETCHCONTENT_QUIET ON)

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
    GXF::multimedia
    ZLIB::ZLIB
)

target_include_directories(volume_renderer INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(volume_renderer PRIVATE HOLOSCAN_MAJOR_VERSION=${holoscan_VERSION_MAJOR})
target_compile_definitions(volume_renderer PRIVATE HOLOSCAN_MINOR_VERSION=${holoscan_VERSION_MINOR})

install(
  TARGETS volume_renderer
  EXPORT holoscan-ops
)

if(HOLOHUB_BUILD_PYTHON)
    add_subdirectory(python)
endif()
